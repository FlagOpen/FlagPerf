"""ResNet50 Pretraining"""
# 标准库
import os
import sys
import time
import random
import numpy as np
from typing import Any, Tuple

# 三方库
import torch
import torch.nn as nn

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(
    CURR_PATH, "../../")))  # add benchmarks directory

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

# 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
from train import trainer_adapter
from train.device import Device
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState

# 这里需要导入dataset, dataloader的相关方法。 这里尽量保证函数的接口一致，实现可以不同。
from dataloaders.dataloader import (
    build_train_dataset,
    build_eval_dataset,
    build_train_dataloader,
    build_eval_dataloader,
    WorkerInitializer,
)

logger = None


def main() -> Tuple[Any, Any]:
    """training entrypoint"""
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(global_module=globals(),
                                           local_module=locals())
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    # get seed
    seed = config.seed if config.seed else 1024
    init_helper.set_seed(seed, model_driver.config.vendor)

    world_size = dist_pytorch.get_world_size()
    config.distributed = world_size > 1 or config.multiprocessing_distributed

    worker_seeds, _ = dist_pytorch.setup_seeds(
        config.seed, config.num_epochs_to_generate_seeds_for, config.device)

    if torch.distributed.is_initialized():
        worker_seed = worker_seeds[torch.distributed.get_rank()]
    else:
        worker_seed = worker_seeds[0]
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitializer.default(worker_seed)

    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    # build dataset, dataloader 【train && validate】
    train_dataset = build_train_dataset(config)
    val_dataset = build_eval_dataset(config)
    train_dataloader = build_train_dataloader(config,
                                              train_dataset,
                                              worker_init_fn=worker_init)
    eval_dataloader = build_eval_dataloader(config, val_dataset)

    # prepare parameters for training
    device = Device.get_device(config)
    criterion = nn.CrossEntropyLoss().to(device)
    evaluator = Evaluator(config, eval_dataloader)

    training_state = TrainingState()
    trainer = Trainer(
        driver=model_driver,
        criterion=criterion,
        adapter=trainer_adapter,
        evaluator=evaluator,
        training_state=training_state,
        device=device,
        config=config,
    )
    training_state.set_trainer(trainer)

    # 设置分布式环境
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    # evaluation
    if not config.do_train:
        init_evaluation_start = time.time()
        training_state.eval_accuracy = evaluator.evaluate(trainer)
        init_evaluation_end = time.time()

        init_evaluation_info = dict(
            eval_acc1=training_state.eval_acc1,
            eval_acc5=training_state.eval_acc5,
            time=init_evaluation_end - init_evaluation_start,
        )
        model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)
        return config, training_state

    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e3

    dist_pytorch.barrier(config.vendor)

    # TRAIN_START事件
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time

    # 训练过程
    epoch = training_state.epoch
    while not training_state.end_training:
        trainer.train_one_epoch(train_dataloader)
        epoch += 1
        training_state.epoch = epoch

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time

    # 训练时长，单位为秒
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e3
    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    e2e_time = time.time() - start
    if config.do_train:
        training_perf = (dist_pytorch.global_batch_size(config) *
                         state.global_steps) / state.raw_train_time

        finished_info = {
            "e2e_time": e2e_time,
            "global_steps": state.global_steps,
            "training_images_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_loss,
            "final_acc1": state.eval_acc1,
            "final_acc5": state.eval_acc5,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
            "num_trained_samples": state.num_trained_samples,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

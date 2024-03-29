# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import sys
import time
from typing import Any, Tuple

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders.dataloader import build_train_dataloader, build_eval_dataloader

logger = None


def main() -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    config.distributed = dist_pytorch.get_world_size() > 1
    # logger
    logger = model_driver.logger

    train_dataloader = build_train_dataloader(config)
    eval_dataloader = build_eval_dataloader(config)

    seed = config.seed

    init_helper.set_seed(seed, model_driver.config.vendor)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    evaluator = Evaluator(config, eval_dataloader)
    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config)
    training_state._trainer = trainer

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init(train_dataloader)
    dist_pytorch.barrier(config.vendor)

    # evaluation统计
    init_evaluation_start = time.time()  # evaluation起始时间，单位为秒

    training_state.acc = evaluator.evaluate(trainer)

    init_evaluation_end = time.time()  # evaluation结束时间，单位为秒

    init_evaluation_info = dict(time=init_evaluation_end -
                                init_evaluation_start)

    model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    if not config.do_train:
        return config, training_state

    model_driver.event(Event.INIT_END)

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    train_start_time = time.time()

    # 训练过程
    epoch = 1
    while not training_state.end_training:
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
        epoch += 1

    # TRAIN_END事件
    training_state.train_time = time.time() - train_start_time
    model_driver.event(Event.TRAIN_END)

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - start
    if config_update.do_train:

        finished_info = {
            "e2e_time": e2e_time,
            "train_time": state.train_time,
            "train_no_eval_time": state.no_eval_time,
            "pure_training_computing_time": state.pure_compute_time,
            "throughput(ips)_raw": state.num_trained_samples / state.train_time,
            "throughput(ips)_no_eval":
            state.num_trained_samples / state.no_eval_time,
            "throughput(ips)_pure_compute":
            state.num_trained_samples / state.pure_compute_time,
            "converged": state.converged,
            "acc": state.acc,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

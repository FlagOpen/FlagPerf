# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Transformer Pretraining"""

import os
import sys
import time
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
import config
from driver.helper import InitHelper
from driver import Event, dist_pytorch

from train.training_state import TrainingState
from train.evaluator import Evaluator
from train.trainer import Trainer
from dataloaders.dataloader import build_dataloader


logger = None


def main():
    global logger
    global config

    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    dist_pytorch.init_dist_training_env(config)
    config.epochs = config.max_epoch
    config.distributed_world_size = int(os.environ.get('WORLD_SIZE',1))
    config.distributed_rank = int(os.environ.get('RANK',0))
    config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist_pytorch.barrier(config.vendor)

    logger = model_driver.logger

    model_driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    # 根据厂商设置全局随机种子
    init_helper.set_seed(config.seed, config.vendor)

    # 加载数据
    train_dataloader, valid_dataloader, test_dataloader = build_dataloader(config)

    # 初始化 训练状态
    training_state = TrainingState()
    # 验证器
    evaluator = Evaluator(config, test_dataloader)
    # 训练器
    trainer = Trainer(driver=model_driver,
                      adapter=None,
                      evaluator=evaluator,
                      training_state=training_state,
                      device=config.device,
                      config=config)
    trainer.init(train_dataloader)
    # 验证原始参数
    init_evaluation_start = time.time()
    training_state.valid_loss = trainer.validate(valid_dataloader)
    init_evaluation_end = time.time()

    init_end_time = logger.previous_log_time

    init_evaluation_info = dict(
        time=init_evaluation_end - init_evaluation_start)
    model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    training_state.init_time = (init_end_time - init_start_time) / 1e+3
    model_driver.event(Event.INIT_END)

    if not config.do_train:
        return training_state

    # 开始训练
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = time.time()
    while training_state.epoch < config.epochs and not training_state.end_training:
        trainer.train_one_epoch(train_dataloader, valid_dataloader)
        training_state.epoch += 1

    model_driver.event(Event.TRAIN_END)
    training_state.raw_train_time = time.time() - raw_train_start_time
    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    e2e_time = time.time() - start
    finished_info = {"e2e_time": e2e_time}
    training_perf = state.total_tokens / state.raw_train_time
    if config.do_train:
        finished_info = {
            "global_steps": state.global_steps,
            "e2e_time": e2e_time,
            "training_tokens_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.valid_loss,
            "final_acc": state.test_bleu,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
            "throughput(ips)_raw": state.total_tokens / state.raw_train_time,
            "throughput(ips)_no_eval": state.total_tokens / state.no_eval_time,
            "throughput(ips)_pure_compute": state.total_tokens / state.pure_compute_time,
        }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

# 标准库
import os
import sys
import time
import argparse

# 三方库
import torch

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper
from train.trainer import Trainer
from train.training_state import TrainingState
from dataloaders.dataloader import build_train_dataset, \
    build_eval_dataset, build_train_dataloader, build_eval_dataloader
from model.model_parser import waveglow_parser, parse_args

logger = None


def main():
    global logger
    global config

    # model config from args
    parser = argparse.ArgumentParser(config.name)
    parser = parse_args(parser)
    parser = waveglow_parser(parser, add_help=False)

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals(), parser)
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # distributed
    config.world_size = dist_pytorch.get_world_size()
    
    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    if config.seed is not None:
        init_helper.set_seed(config.seed + config.local_rank,
                             model_driver.config.vendor)

    #创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    trainer = Trainer(training_state=training_state, config=config)

    # 设置分布式环境, trainer init()
    dist_pytorch.barrier(config.vendor)
    trainer.init()
    dist_pytorch.barrier(config.vendor)

    # do evaluation
    if not config.do_train:
        return config, training_state

    trainset = build_train_dataset(config)
    train_loader = build_train_dataloader(trainset, config)
    valset = build_eval_dataset(config)
    val_loader = build_eval_dataloader(valset, config)

    start_epoch = [0]
    start_epoch = start_epoch[0]
    iteration = 0
    train_epoch_items_per_sec = 0.0
    val_loss = 0.0
    num_iters = 0
    epoch = start_epoch

    #init 统计
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time  # init结束时间，单位为ms
    training_state.init_time = (init_end_time -
                                init_start_time) / 1e+3  # 初始化时长，单位为秒
    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time  # 训练起始时间，单位为ms

    while epoch >= start_epoch and epoch < config.epochs and not training_state.end_training:
        training_state.epoch = epoch
        train_epoch_items_per_sec, val_items_per_sec, val_loss, num_iters = trainer.train_one_epoch(
            epoch, train_loader, val_loader, config, iteration,
            train_epoch_items_per_sec, val_loss)
        epoch += 1

    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time  # 训练结束时间，单位为ms
    # 训练时长，单位为秒
    torch.cuda.synchronize()
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3
    training_state.train_items_per_sec = (train_epoch_items_per_sec /
                                          num_iters if num_iters > 0 else 0.0)
    training_state.val_items_per_sec = val_items_per_sec
    training_state.val_loss = val_loss

    return config, training_state


if __name__ == '__main__':
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)

    # 训练信息写日志
    e2e_time = time.time() - state.train_start_timestamp
    if config_update.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_items_per_second": state.train_items_per_sec,
            "val_items_per_sec": state.val_items_per_sec,
            "converged": state.converged,
            "final_loss": state.val_loss,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

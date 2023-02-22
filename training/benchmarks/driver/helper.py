# Copyright © 2022 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import os
import random
import time
import numpy as np
import torch
from . import dist_pytorch as distributed
from driver import perf_logger, Driver
import driver


class InitHelper:
    """
    定义run_pretrain中的通用逻辑
    """

    def __init__(self, config: object) -> None:
        self.config = config

    def get_logger(self) -> perf_logger.PerfLogger:
        """get logger for FlagPerf"""
        return perf_logger.PerfLogger.get_default_logger(
            rank=self.config.local_rank)

    def get_local_rank(self) -> int:
        """get local rank"""
        if self.config.use_env and 'LOCAL_RANK' in os.environ:
            return int(os.environ['LOCAL_RANK'])
        return 0

    def set_seed(self, seed: int, vendor: str):
        """set seed"""
        random.seed(seed)
        np.random.seed(seed)
        lower_vendor = vendor.lower()
        if lower_vendor == "nvidia":
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # 其他厂商设置seed，在此扩展
            pass

    def init_driver(self, name: str, device: str = None) -> Driver:
        """
        params:
            name: driver name
            device: cuda - nvidia
                    xpu - kulunxin
                    iluvatar - tianshu

        """
        """
        1. init driver object
        2. setup_config
        3. setup_modules
        """
        config = self.config
        model_driver = Driver(config, config.mutable_params)
        model_driver.setup_config(argparse.ArgumentParser(name))
        model_driver.setup_modules(driver, globals(), locals())
        return model_driver


def get_finished_info(start_time: int, state: object, do_train: bool,
                      global_batch_size: int) -> dict:
    """
    :param start_time start timestamp for training
    :param state training state
    return train state info
    """
    e2e_time = time.time() - start_time
    finished_info = {"e2e_time": e2e_time}

    if do_train:
        training_perf = (global_batch_size *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_accuracy": state.eval_accuracy,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    return finished_info
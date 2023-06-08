# Copyright © 2022 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import os
import random
import numpy as np
import torch
from driver import perf_logger, Driver, check


class InitHelper:
    """
    定义run_pretrain中的通用逻辑
    """

    def __init__(self, config: object) -> None:
        self.config = config

    def init_driver(self, global_module, local_module, parser=None) -> Driver:
        """
        params:
            name: model name
        """
        config = self.config
        model_driver = Driver(config, config.mutable_params)
        parser = argparse.ArgumentParser(config.name) if parser is None else parser
        model_driver.setup_config(parser)
        model_driver.setup_modules(global_module, local_module)
        check.check_config(model_driver.config)
        self.update_local_rank()
        return model_driver

    def get_logger(self) -> perf_logger.PerfLogger:
        """get logger for FlagPerf"""
        return perf_logger.PerfLogger.get_default_logger(
            rank=self.config.local_rank)

    def update_local_rank(self) -> int:
        """set local rank"""
        self.config.local_rank = int(os.getenv("LOCAL_RANK", 0))

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
        elif lower_vendor == "iluvatar":
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        elif lower_vendor == "kunlunxin":
            torch.manual_seed(seed)
        else:
            # TODO 其他厂商设置seed，在此扩展
            pass

# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import os.path as ospath
from .dist_pytorch import global_batch_size


def get_config_arg(config, name):
    if hasattr(config, name):
        value = getattr(config, name)
        if value is not None:
            return value

    if name in os.environ:
        return os.environ[name]

    return None


def check_config(config):
    data_dir = get_config_arg(config, "data_dir")
    if data_dir is None:
        raise ValueError("Invalid data_dir, should be given a path.")
    if not ospath.isdir(data_dir):
        raise ValueError(f"data_dir '{data_dir}' not exists.")
    config.data_dir = data_dir

    train_data = get_config_arg(config, "train_data")
    if train_data is not None:
        config.train_data = ospath.join(data_dir, train_data)

    eval_data = get_config_arg(config, "eval_data")
    if eval_data is not None:
        config.eval_data = ospath.join(data_dir, eval_data)

    init_checkpoint = get_config_arg(config, "init_checkpoint")
    if init_checkpoint is not None:
        config.init_checkpoint = ospath.join(data_dir, config.init_checkpoint)

    if config.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(config.gradient_accumulation_steps))
# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import config


def convert_model(model: nn.Module) -> nn.Module:
    """convert_model"""
    return model

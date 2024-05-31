# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch

import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import config
from driver.dist_pytorch import main_proc_print


def convert_model(model: nn.Module) -> nn.Module:
    return model


def model_to_fp16(model: nn.Module) -> nn.Module:
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[config.local_rank])
    return model


def create_grad_scaler():
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    return scaler

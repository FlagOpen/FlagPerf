# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from torch.optim import Optimizer
from torch import nn, Tensor

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


def create_grad_scaler():
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp,
                                       growth_interval=int(1e9))
    return scaler


def backward(loss: Tensor, optimizer: Optimizer):
    """backward pass"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

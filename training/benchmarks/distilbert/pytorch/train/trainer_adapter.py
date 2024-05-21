# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import config

def convert_model(model: nn.Module) -> nn.Module:
    """convert_model"""
    return model


def model_to_ddp(config, model: nn.Module) -> nn.Module:
    use_ddp = dist.is_initialized()
    if use_ddp:
        model = DistributedDataParallel(
            model,
            device_ids=[config.local_rank])

    return model


def backward(config, step: int, loss: torch.Tensor, optimizer, **kwarg):
    if config.gradient_accumulation_steps > 1:
        loss = loss / config.gradient_accumulation_steps

    loss.backward()

    if step % config.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
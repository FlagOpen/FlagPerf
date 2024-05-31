# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

from typing import Tuple

import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import config
from optimizer import get_megatron_optimizer

GPT_MODEL = torch.nn.Module

def convert_model(config, model: GPT_MODEL) -> GPT_MODEL:
    return model

def create_optimizer(config, model: GPT_MODEL) -> Optimizer:
    return get_megatron_optimizer(model)


def model_to_fp16(config, model: GPT_MODEL,
                  optimizer: Optimizer) -> Tuple[GPT_MODEL, Optimizer]:
    if config.fp16:
        model.half()
    return model, optimizer


def model_to_ddp(config, model: GPT_MODEL) -> GPT_MODEL:
    use_ddp = dist.is_initialized()

    if use_ddp:
        if config.DDP_impl == 'native':
            model = NativeDDP(
                model,
                device_ids=[config.local_rank])
        else:
            assert False, "Invalid DDP type"
    return model


def backward(step: int,
             loss: torch.Tensor,
             optimizer: Optimizer,
             lr_scheduler):
    if config.fp16:
        optimizer.backward(loss)
    else:
        loss.backward()

    if step % config.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        increment = config.train_batch_size * config.n_device * config.gradient_accumulation_steps
        lr_scheduler.step(increment)

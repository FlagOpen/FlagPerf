# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from torch.optim import Optimizer
from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print, is_dist_avail_and_initialized
from torch.nn.parallel import DistributedDataParallel as DDP

import config


def create_model(args, device, device_mapping, feature_spec):
    pass

def create_mlp_optimizer(model, args):
    pass
  
def create_grad_scaler(args):
    pass

def create_scheduler(args, mlp_optimizer, embedding_optimizer):
    pass

def get_trainer_wrapper(model, config, embedding_optimizer, mlp_optimizer,
                           loss_fn, grad_scaler):
    return model


def convert_model(model: nn.Module) -> nn.Module:
    return model


def model_to_fp16(model: nn.Module) -> nn.Module:
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[config.local_rank])
    return model


    


def backward(loss: Tensor, optimizer: Optimizer):
    """backward pass"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver.dist_pytorch import main_proc_print


def create_optimizer(model, args, loss_module):
    pass


def create_clip_grad():
    pass


def create_grad_scaler(args):
    pass


def convert_model(model: nn.Module) -> nn.Module:
    return model


def model_to_fp16(model: nn.Module, args) -> nn.Module:
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        main_proc_print(" > use fp16...")
        model.to(torch.bfloat16)
    return model


def model_to_ddp(model: nn.Module, args) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank])
    return model

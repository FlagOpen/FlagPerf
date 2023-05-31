# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import sys
import torch

from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))


def model_to_fp16(model: nn.Module, args):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        dist.main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model: nn.Module, args) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = DDP(model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank)
    return model


def create_grad_scaler(args):
    return torch.cuda.amp.GradScaler(enabled=args.amp)


def calculate_loss(model, args, criterion, x, y):
    # AMP upstream autocast
    with torch.cuda.amp.autocast(enabled=args.amp):
        y_pred = model(x)
        loss = criterion(y_pred, y)
    return loss

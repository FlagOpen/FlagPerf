import os
from port_for import is_available
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import config

from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel as DDP


def convert_model(model: nn.Module) -> nn.Module:
    return model


def create_optimizer(model, args):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def model_to_fp16(model):
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
    return None


def backward(step: int, loss: torch.Tensor, optimizer: Optimizer):
    loss.backward()
    update_step = step % config.gradient_accumulation_steps == 0
    if update_step:
        optimizer.step()
        optimizer.zero_grad()

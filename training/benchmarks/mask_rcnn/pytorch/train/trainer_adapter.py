import os
import sys
import torch

import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import config

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver.dist_pytorch import main_proc_print


def convert_model(model: nn.Module) -> nn.Module:
    """convert_model"""
    return model


def model_to_fp16(model: nn.Module) -> nn.Module:
    """model_to_fp16"""
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    """model_to_ddp"""
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[config.local_rank])
    return model


def create_grad_scaler():
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    return scaler


def sum_loss(model, scaler, images, targets):
    # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
    return loss_dict, losses
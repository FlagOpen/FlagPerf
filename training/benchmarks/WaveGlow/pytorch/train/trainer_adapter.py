import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from driver import dist_pytorch


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver.dist_pytorch import main_proc_print


def convert_model(model):
    """convert_model"""
    return model


def model_to_fp16(model, config):
    """model_to_fp16"""
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model, config):
    """model_to_ddp"""
    if dist_pytorch.is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[config.local_rank])
    return model


def create_grad_scaler(args):
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    return scaler

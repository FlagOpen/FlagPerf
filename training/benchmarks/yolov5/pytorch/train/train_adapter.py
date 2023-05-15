import torch
import torch.distributed as dist
from torch.optim import Optimizer
import config
import os

from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.autobatch import check_train_batch_size
from utils.general import (check_amp, check_img_size)
from utils.torch_utils import smart_optimizer

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def convert_model(model: nn.Module) -> nn.Module:
    return model

def create_optimizer(model, args):

    amp = check_amp(model)
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        # loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    
    from schedulers import hyp
    
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, args.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

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

import torch
from torch.optim import Optimizer
import torch.distributed as dist
import config
from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print
from torch.nn.parallel import DistributedDataParallel as DDP


def convert_model(model: nn.Module) -> nn.Module:
    """convert model"""
    return model


def create_optimizer(model) -> Optimizer:
    """create optimizer"""
    optimizer = torch.optim.SGD(
        model.parameters(),
        config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay_rate,
    )
    return optimizer


def model_to_fp16(model):
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
    """create_grad_scaler"""
    return None


def backward(step: int, loss: Tensor, optimizer: Optimizer):
    """backward"""
    # compute gradient and do SGD step
    loss.backward()
    update_step = step % config.gradient_accumulation_steps == 0
    if update_step:
        optimizer.step()
        optimizer.zero_grad()

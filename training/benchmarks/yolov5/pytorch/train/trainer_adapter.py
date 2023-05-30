import torch
import torch.distributed as dist
from torch.optim import Optimizer
import config

from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler


def convert_model(model: nn.Module) -> nn.Module:
    """convert model"""
    return model

def create_optimizer(nbs, batch_size, model, opt, hyp):
    # Optimizer
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.optimizer == 'Adam':
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2
    
    return optimizer
    
    
def model_to_fp16(model):
    """model_to_fp16"""
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model

def model_to_ddp(model: nn.Module) -> nn.Module: 
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[config.local_rank],output_device=[config.local_rank])
    return model

# TODO
def backward(step: int, loss: Tensor, optimizer: Optimizer):
    """backward"""
    pass



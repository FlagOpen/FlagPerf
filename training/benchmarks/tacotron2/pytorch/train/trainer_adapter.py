import os
import sys

from torch.optim import Optimizer
from torch import nn, Tensor
from typing import Tuple

import optimizers
try:
    from apex.optimizers import FusedAdam as Adam
except ImportError:
    from torch.optim import AdamW as Adam
# from optimizers import FP16_Optimizer, get_optimizer_param_groups

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver.dist_pytorch import main_proc_print


def convert_model(model: nn.Module) -> nn.Module:
    return model


def create_optimizer(model, args):
    param_groups = get_optimizer_param_groups(model)
    optimizer = Adam(param_groups,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     betas=(args.adam_beta1, args.adam_beta2),
                     eps=args.adam_eps)
    main_proc_print(f'Optimizer = {optimizer.__class__.__name__}')
    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis
                                   })

    return optimizer


def model_to_fp16(model: nn.Module,
                  optimizer: Optimizer) -> Tuple[nn.Module, Optimizer]:
    return model, optimizer


def model_to_ddp(model: nn.Module) -> nn.Module:
    return model


def create_grad_scaler():
    return None


def backward(step: int, loss: Tensor, optimizer, **kwarg):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

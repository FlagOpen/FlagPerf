import os
import torch
import config

from torch import nn, Tensor

from .converter import convert_model as _convert_model
from optimizers import FP16_Optimizer, get_optimizer_param_groups
from utils import main_proc_print
from apex.optimizers import FusedAdam as Adam
from typing import Tuple
from model.models.modeling import FP16_Module
from utils import PyTorchDistributedDataParallel as TorchDDP

from optimizers.loss_scaler import DynamicLossScaler

clip_grad_norm = torch.nn.utils.clip_grad_norm_


def convert_model(model: torch.nn.Module) -> torch.nn.Module:
    return _convert_model(model, config)


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


def model_to_fp16(model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if config.fp16:
        model = FP16_Module(model)
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    i = torch.cuda.current_device()
    model = TorchDDP(model, device_ids=[i], output_device=i)
    return model


def create_grad_scaler():
    return None


def backward(step, lm_loss, reduced_loss, optimizer, lr_scheduler, model):
    args = config

    if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
        backward_step(optimizer, model, lm_loss, args)
        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            if not (args.fp16 and optimizer.overflow):
                lr_scheduler.step()
            optimizer.zero_grad()

    else:
        main_proc_print("Found NaN loss, skip backward")
    return reduced_loss


def backward_step(optimizer, model, lm_loss, args):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss

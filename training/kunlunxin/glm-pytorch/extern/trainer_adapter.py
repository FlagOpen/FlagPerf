import torch
from torch import nn
import torch.distributed as dist

import torch_xmlir

import config
from optimizers import get_optimizer_param_groups
from optimizers.loss_scaler import DynamicLossScaler
from driver.dist_pytorch import main_proc_print

from driver.dist_pytorch import PyTorchDistributedDataParallel as TorchDDP

clip_grad_norm = torch.nn.utils.clip_grad_norm_

from .converter import convert_model as _convert_model


def convert_model(model: torch.nn.Module) -> torch.nn.Module:
    return _convert_model(model, config)


def model_to_fp16(model):
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = TorchDDP(model)
    return model


def backward(step, lm_loss, reduced_loss, optimizer, lr_scheduler, model):
    args = config

    if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
        backward_step(optimizer, model, lm_loss, args)
        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    else:
        main_proc_print("Found NaN loss, skip backward")

    torch_xmlir.xpu.empty_cache()
    return reduced_loss


def backward_step(optimizer, model, lm_loss, args):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    loss.backward()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        clip_grad_norm(model.parameters(), args.clip_grad)

    return lm_loss

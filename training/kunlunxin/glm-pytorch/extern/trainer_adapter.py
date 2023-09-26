import torch
from torch import nn
import torch.distributed as dist

import config
from optimizers import get_optimizer_param_groups
from optimizers.loss_scaler import DynamicLossScaler
from driver.dist_pytorch import main_proc_print

import torch_xmlir
import torch_xmlir.core.xpu_model as xm
from torch_xmlir.optimizer import AdamW as Adam
from torch_xmlir.nn.clip_grad import clip_grad_norm
from torch_xmlir.distributed import DistributedDataParallel as XPUDDP

from .converter import convert_model as _convert_model


class XPUTorchDDP(XPUDDP):

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


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
    return optimizer


def model_to_fp16(model):
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = XPUTorchDDP(model)
    return model


def create_grad_scaler():
    return None


def backward(step, lm_loss, reduced_loss, optimizer, lr_scheduler, model):
    args = config

    def _clip_grad():
        if args.clip_grad > 0:
            clip_grad_norm(model.parameters(), args.clip_grad)

    lm_loss.backward()
    if step % args.gradient_accumulation_steps == 0:
        allreduce_grads = reversed(
            [p.grad.data for p in model.parameters() if p.grad is not None])
        xm.optimizer_step(optimizer,
                          barrier=True,
                          post_allreduce_hook=_clip_grad,
                          allreduce_average=True,
                          allreduce_grads=allreduce_grads)
        lr_scheduler.step()

    if DynamicLossScaler._has_inf_or_nan(reduced_loss):
        main_proc_print("Found NaN loss, skip backward")

    torch_xmlir.xpu.empty_cache()
    return reduced_loss

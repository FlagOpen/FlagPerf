import torch
import torch.distributed as dist
from torch.optim import Optimizer

from torch import nn, Tensor
from driver.dist_pytorch import main_proc_print
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
from train import utils
import os
import sys

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import dist_pytorch


def convert_model(args, model: nn.Module) -> nn.Module:
    if dist_pytorch.is_dist_avail_and_initialized() and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def create_optimizer(args, model):
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
                "class_token", "position_embedding",
                "relative_position_bias_table"
        ]:
            custom_keys_weight_decay.append(
                (key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters,
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        eps=0.0316,
                                        alpha=0.9)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )
    return optimizer


def model_to_fp16(args, model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(args, model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank])
    return model


def create_grad_scaler(args):
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    return scaler


def backward(args, step: int, epoch: int, loss: torch.Tensor, model: nn.Module,
             optimizer: Optimizer, scaler):
    if scaler is not None:
        scaler.scale(loss).backward()
        if step % args.gradient_accumulation_steps == 0:
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
    else:
        loss.backward()
        if step % args.gradient_accumulation_steps == 0:
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

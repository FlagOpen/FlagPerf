import torch
from apex.optimizers import FusedAdam as Adam
from apex.contrib.clip_grad import clip_grad_norm_

def create_optimizer(model, args, loss_module):
    optimizer = Adam((*model.parameters(), *loss_module.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
    return optimizer


def create_clip_grad():
    return clip_grad_norm_


def create_grad_scaler(args):
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    return scaler
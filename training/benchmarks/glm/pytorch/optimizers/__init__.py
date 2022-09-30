from torch.optim import AdamW as Adam
import torch

from .fp16_optimizer import FP16_Optimizer
from utils import PyTorchDistributedDataParallel as TorchDDP
from model.models.modeling import FP16_Module
from utils import print_rank_0

def create_optimizer(model, args):
    param_groups = get_optimizer_param_groups(model)
    # print("weight decay:",args.weight_decay)
    # print("adam_beta1:",args.adam_beta1)
    # print("adam_beta2:",args.adam_beta2)
    # print("adam_eps:",args.adam_eps)
    # exit()
    optimizer = Adam(param_groups,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     betas=(args.adam_beta1, args.adam_beta2),
                     eps=args.adam_eps)
    print_rank_0(f'Optimizer = {optimizer.__class__.__name__}')
    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (TorchDDP, FP16_Module)):
        model = model.module
    param_groups = glm_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        # print('## param_group', len(param_group['params']))
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False
    # print("model_parallel:", param.model_parallel)
    return param_groups


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n == 'bias'])

    return weight_decay_params, no_weight_decay_params

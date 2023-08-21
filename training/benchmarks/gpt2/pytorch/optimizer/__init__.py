# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from torch.optim import SGD
from torch.optim import AdamW as Adam

from optimizer.distrib_optimizer import DistributedOptimizer
from optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
import config

def get_param_groups(module,
                     no_weight_decay_cond,
                     scale_lr_cond,
                     lr_mult):
    """creates param groups based on weight decay condition (regularized vs non regularized)
       and learning rate scale condition (args.lr vs lr_mult * args.lr)
       scale_lr_cond is used during finetuning where head of the network requires a scaled
       version of the base learning rate.
    """
    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        if no_weight_decay_cond is not None:
            no_wd = no_weight_decay_cond(name, param)
        else:
            # do not regularize biases nor Norm parameters
            no_wd = name.endswith(".bias") or len(param.shape) == 1

        if scale_lr_cond is not None:
            scale_lr = scale_lr_cond(name, param)
        else:
            scale_lr = False

        if not no_wd and not scale_lr:
            wd_no_scale_lr.append(param)
        elif not no_wd and scale_lr:
            wd_scale_lr.append(param)
        elif no_wd and not scale_lr:
            no_wd_no_scale_lr.append(param)
        else:
            no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(wd_scale_lr):
        param_groups.append({'params': wd_scale_lr, 'wd_mult': 1.0, 'lr_mult': lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({'params': no_wd_no_scale_lr, 'wd_mult': 0.0, 'lr_mult': 1.0})
    if len(no_wd_scale_lr):
        param_groups.append({'params': no_wd_scale_lr, 'wd_mult': 0.0, 'lr_mult': lr_mult})

    return param_groups

def get_megatron_optimizer(model,
                           no_weight_decay_cond=None,
                           scale_lr_cond=None,
                           lr_mult=1.0):
    # Base optimizer.
    param_groups = get_param_groups(model,
                                    no_weight_decay_cond,
                                    scale_lr_cond,
                                    lr_mult)

    if config.optimizer == 'adam':
        optimizer = Adam(param_groups,
                         lr=config.lr,
                         weight_decay=config.weight_decay,
                         betas=(config.adam_beta1, config.adam_beta2),
                         eps=config.adam_eps)
    elif config.optimizer == 'sgd':
        optimizer = SGD(param_groups,
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                        momentum=config.sgd_momentum)
    else:
        raise Exception('{} optimizer is not supported.'.format(
            config.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if config.DDP_impl == 'local':
        params_have_main_grad = True

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis)

        # Megatron optimizer.
        opt_ty = DistributedOptimizer \
            if config.use_distributed_optimizer else \
            Float16OptimizerWithFloat16Params
        return opt_ty(optimizer,
                      config.clip_grad,
                      params_have_main_grad,
                      config.use_contiguous_buffers_in_local_ddp,
                      config.fp16,
                      config.bf16,
                      config.params_dtype,
                      grad_scaler,
                      model)

    # FP32.
    
    return FP32Optimizer(optimizer, config.clip_grad,
                         params_have_main_grad,
                         config.use_contiguous_buffers_in_local_ddp,
                         model)

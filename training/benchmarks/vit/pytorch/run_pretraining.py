#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import sys

import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import ApexScaler, NativeScaler
from timm.data import resolve_data_config, Mixup, FastCollateMixup, AugMixDataset


from train.trainer import Trainer
from dataloaders.dataloader import build_train_dataset, \
    build_eval_dataset, build_train_dataloader, build_eval_dataloader
from schedulers import create_scheduler
from driver import Event, dist_pytorch
from driver.helper import InitHelper

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

def main():
    import config
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    
    utils.setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    config.prefetcher = not config.no_prefetcher
    device = utils.init_distributed_device(config)
    if config.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {config.rank}, total {config.world_size}, device {config.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({config.device}).')
    assert config.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if config.amp:
        if config.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert config.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert config.amp_dtype in ('float16', 'bfloat16')
        if config.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(config.seed, config.rank)

    if config.fuser:
        utils.set_jit_fuser(config.fuser)
    if config.fast_norm:
        set_fast_norm()
        
    trainer = Trainer(device=device, args=config)
    trainer.init()
    
    data_config = resolve_data_config(vars(config), model=trainer.model, verbose=utils.is_primary(config))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        trainer.model, trainer.optimizer = amp.initialize(trainer.model, trainer.optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(config):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda':
            loss_scaler = NativeScaler()
        if utils.is_primary(config):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(config):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if config.resume:
        resume_epoch = resume_checkpoint(
            trianer.model,
            config.resume,
            optimizer=None if config.no_resume_opt else trainer.optimizer,
            loss_scaler=None if config.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(config),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if config.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            trainer.model, decay=config.model_ema_decay, device='cpu' if config.model_ema_force_cpu else None)
        if config.resume:
            load_checkpoint(model_ema.module, config.resume, use_ema=True)

    # setup distributed training
    if config.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(config):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            trainer.model = ApexDDP(trainer.model, delay_allreduce=True)
        else:
            if utils.is_primary(config):
                _logger.info("Using native Torch DistributedDataParallel.")
            trainer.model = NativeDDP(trainer.model, device_ids=[device], broadcast_buffers=not config.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # create the train and eval datasets
    if config.data and not config.data_dir:
        config.data_dir = config.data
    dataset_train = build_train_dataset(config)
    dataset_eval = build_eval_dataset(config)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    if trainer.mixup_active:
        mixup_args = dict(
            mixup_alpha=config.mixup,
            cutmix_alpha=config.cutmix,
            cutmix_minmax=config.cutmix_minmax,
            prob=config.mixup_prob,
            switch_prob=config.mixup_switch_prob,
            mode=config.mixup_mode,
            label_smoothing=config.smoothing,
            num_classes=config.num_classes
        )
        if config.prefetcher:
            assert not trainer.num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if trainer.num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=trainer.num_aug_splits)

    loader_train = build_train_dataloader(dataset_train, data_config, trainer.num_aug_splits, collate_fn, device, config)
    loader_eval = build_eval_dataloader(dataset_eval, data_config, device, config)

    # setup checkpoint saver and eval metric tracking
    eval_metric = config.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(config):
        if config.experiment:
            exp_name = config.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(config.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(config.output if config.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=trainer.model,
            optimizer=trainer.optimizer,
            args=config,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=config.checkpoint_hist
        )

    if utils.is_primary(config) and config.log_wandb:
        if has_wandb:
            wandb.init(project=config.experiment, config=config)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler(
        trainer.optimizer,
        updates_per_epoch=updates_per_epoch,
        args=config,
    )
    start_epoch = 0
    if config.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = config.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if config.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(config):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif config.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = trainer.train_one_epoch(
                epoch,
                loader_train,
                # train_loss_fn,
                # args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            if config.distributed and config.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(config):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, config.world_size, config.dist_bn == 'reduce')

            eval_metrics = trainer.validate(
                trainer.model,
                loader_eval,
                trainer.validate_loss_fn,
                config,
                device=device,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not config.model_ema_force_cpu:
                if config.distributed and config.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, config.world_size, config.dist_bn == 'reduce')

                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    trainer.validate_loss_fn,
                    config,
                    device=device,
                    amp_autocast=amp_autocast,
                    log_suffix=' (EMA)',
                )
                eval_metrics = ema_eval_metrics

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in trainer.optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=config.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

if __name__ == '__main__':
    main()

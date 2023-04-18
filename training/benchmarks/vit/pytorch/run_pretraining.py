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
from timm.layers import set_fast_norm
from timm.models import safe_model_name, resume_checkpoint, load_checkpoint
from timm.utils import ApexScaler, NativeScaler
from timm.data import resolve_data_config, Mixup, FastCollateMixup, AugMixDataset


from train.trainer import Trainer
from dataloaders.dataloader import build_train_dataset, \
    build_eval_dataset, build_train_dataloader, build_eval_dataloader
from schedulers import create_scheduler
from driver import Event, dist_pytorch
from driver.helper import InitHelper
from train import trainer_adapter
from train.training_state import TrainingState


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

logger = None

def main():
    import config
    global logger
    
    config.prefetcher = not config.no_prefetcher
    if not config.train_batch_size:
        config.train_batch_size = config.batch_size
    
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    
    utils.setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dist_pytorch.init_dist_training_env(config)
    
    model_driver.event(Event.INIT_START)
    
    logger = model_driver.logger
    init_start_time = logger.previous_log_time
    
    init_helper.set_seed(config.seed, config.vendor)

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

    if config.fuser:
        utils.set_jit_fuser(config.fuser)
    if config.fast_norm:
        set_fast_norm()
        
    training_state = TrainingState()
    trainer = Trainer(driver=model_driver,
                      adapter=trainer_adapter,
                      training_state=training_state,
                      device=config.device,
                      args=config)
    training_state._trainer = trainer
    
    trainer.init()
    
    data_config = resolve_data_config(vars(config), model=trainer.model, verbose=dist_pytorch.is_main_process())

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert config.device.type == 'cuda'
        trainer.model, trainer.optimizer = amp.initialize(trainer.model, trainer.optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if dist_pytorch.is_main_process():
            print('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=config.device.type, dtype=amp_dtype)
        if config.device.type == 'cuda':
            loss_scaler = NativeScaler()
        if dist_pytorch.is_main_process():
            print('Using native Torch AMP. Training in mixed precision.')
    else:
        if dist_pytorch.is_main_process():
            print('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if config.resume:
        resume_epoch = resume_checkpoint(
            trianer.model,
            config.resume,
            optimizer=None if config.no_resume_opt else trainer.optimizer,
            loss_scaler=None if config.no_resume_opt else loss_scaler,
            log_info=dist_pytorch.is_main_process(),
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
            if dist_pytorch.is_main_process():
                print("Using NVIDIA APEX DistributedDataParallel.")
            trainer.model = ApexDDP(trainer.model, delay_allreduce=True)
        else:
            if dist_pytorch.is_main_process():
                print("Using native Torch DistributedDataParallel.")
            trainer.model = NativeDDP(trainer.model, device_ids=[config.device], broadcast_buffers=not config.no_ddp_bb)
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

    loader_train = build_train_dataloader(dataset_train, data_config, trainer.num_aug_splits, collate_fn, config.device, config)
    loader_eval = build_eval_dataloader(dataset_eval, data_config, config.device, config)

    # setup checkpoint saver and eval metric tracking
    eval_metric = config.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if dist_pytorch.is_main_process():
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

    if dist_pytorch.is_main_process() and config.log_wandb:
        if has_wandb:
            wandb.init(project=config.experiment, config=config)

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler(
        trainer.optimizer,
        updates_per_epoch=updates_per_epoch,
        args=config,
    )
    
    if not config.do_train:
        return config, training_state
    
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3
    
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time
    
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

    if dist_pytorch.is_main_process():
        print(
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
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            if config.distributed and config.dist_bn in ('broadcast', 'reduce'):
                if dist_pytorch.is_main_process():
                    print("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, config.world_size, config.dist_bn == 'reduce')
                
            eval_start = time.time()
            eval_metrics, training_state.eval_loss, training_state.eval_acc1, training_state.eval_acc5 = trainer.validate(
                trainer.model,
                loader_eval,
                amp_autocast=amp_autocast,
            )
            eval_end = time.time()
            eval_result = dict(global_steps=training_state.global_steps,
                                eval_loss=training_state.eval_loss,
                                eval_acc1=training_state.eval_acc1,
                                eval_acc5=training_state.eval_acc5,
                                time=eval_end - eval_start)

            if eval_result is not None:
                model_driver.event(Event.EVALUATE, eval_result)

            if model_ema is not None and not config.model_ema_force_cpu:
                if config.distributed and config.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, config.world_size, config.dist_bn == 'reduce')

                ema_eval_metrics = trainer.validate(
                    model_ema.module,
                    loader_eval,
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

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time
    
    training_state.raw_train_time = (raw_train_end_time -
                                    raw_train_start_time) / 1e+3
    
    if best_metric is not None:
        print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    return config, training_state

if __name__ == '__main__':
    start = time.time()
    config, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)
    global_batch_size = dist_pytorch.global_batch_size(config)
    e2e_time = time.time() - start
    finished_info = {"e2e_time": e2e_time} 
    if config.do_train:
        training_perf = (global_batch_size *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_images_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_loss,
            "final_acc1": state.eval_acc1,
            "final_acc5": state.eval_acc5,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
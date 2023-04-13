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
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

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

# # The first arg parser parses out only the --config argument, this argument is used to
# # load a yaml file containing key-values that override the defaults for the main parser below
# config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
# parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
#                     help='YAML config file specifying default arguments')


# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# # Dataset parameters
# group = parser.add_argument_group('Dataset parameters')

# # device
# group.add_argument('--device', default='cpu', type=str,
#                    help='device of model to train, one of(cpu, cuda, xla) (default: "cpu")')

# # Keep this argument outside the dataset group because it is positional.
# parser.add_argument('data', nargs='?', metavar='DIR', const=None,
#                     help='path to dataset (positional is *deprecated*, use --data-dir)')
# parser.add_argument('--data-dir', metavar='DIR',
#                     help='path to dataset (root dir)')
# parser.add_argument('--dataset', metavar='NAME', default='',
#                     help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
# group.add_argument('--train-split', metavar='NAME', default='train',
#                    help='dataset train split (default: train)')
# group.add_argument('--val-split', metavar='NAME', default='validation',
#                    help='dataset validation split (default: validation)')
# group.add_argument('--dataset-download', action='store_true', default=False,
#                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
# group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
#                    help='path to class to idx mapping file (default: "")')

# # Model parameters
# group = parser.add_argument_group('Model parameters')
# group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
#                    help='Name of model to train (default: "resnet50")')
# group.add_argument('--pretrained', action='store_true', default=False,
#                    help='Start with pretrained version of specified network (if avail)')
# group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
#                    help='Initialize model from this checkpoint (default: none)')
# group.add_argument('--resume', default='', type=str, metavar='PATH',
#                    help='Resume full model and optimizer state from checkpoint (default: none)')
# group.add_argument('--no-resume-opt', action='store_true', default=False,
#                    help='prevent resume of optimizer state when resuming model')
# group.add_argument('--num-classes', type=int, default=None, metavar='N',
#                    help='number of label classes (Model default if None)')
# group.add_argument('--gp', default=None, type=str, metavar='POOL',
#                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
# group.add_argument('--img-size', type=int, default=None, metavar='N',
#                    help='Image size (default: None => model default)')
# group.add_argument('--in-chans', type=int, default=None, metavar='N',
#                    help='Image input channels (default: None => 3)')
# group.add_argument('--input-size', default=None, nargs=3, type=int,
#                    metavar='N N N',
#                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
# group.add_argument('--crop-pct', default=None, type=float,
#                    metavar='N', help='Input image center crop percent (for validation only)')
# group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
#                    help='Override mean pixel value of dataset')
# group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
#                    help='Override std deviation of dataset')
# group.add_argument('--interpolation', default='', type=str, metavar='NAME',
#                    help='Image resize interpolation type (overrides model)')
# group.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
#                    help='Input batch size for training (default: 128)')
# group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
#                    help='Validation batch size override (default: None)')
# group.add_argument('--channels-last', action='store_true', default=False,
#                    help='Use channels_last memory layout')
# group.add_argument('--fuser', default='', type=str,
#                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
# group.add_argument('--grad-checkpointing', action='store_true', default=False,
#                    help='Enable gradient checkpointing through model blocks/stages')
# group.add_argument('--fast-norm', default=False, action='store_true',
#                    help='enable experimental fast-norm')
# group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# scripting_group = group.add_mutually_exclusive_group()
# scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
#                              help='torch.jit.script the full model')
# scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
#                              help="Enable compilation w/ specified backend (default: inductor).")
# scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
#                              help="Enable AOT Autograd support.")

# # Optimizer parameters
# group = parser.add_argument_group('Optimizer parameters')
# group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
#                    help='Optimizer (default: "sgd")')
# group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
#                    help='Optimizer Epsilon (default: None, use opt default)')
# group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
#                    help='Optimizer Betas (default: None, use opt default)')
# group.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                    help='Optimizer momentum (default: 0.9)')
# group.add_argument('--weight-decay', type=float, default=2e-5,
#                    help='weight decay (default: 2e-5)')
# group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
#                    help='Clip gradient norm (default: None, no clipping)')
# group.add_argument('--clip-mode', type=str, default='norm',
#                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
# group.add_argument('--layer-decay', type=float, default=None,
#                    help='layer-wise learning rate decay (default: None)')
# group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# # Learning rate schedule parameters
# group = parser.add_argument_group('Learning rate schedule parameters')
# group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
#                    help='LR scheduler (default: "step"')
# group.add_argument('--sched-on-updates', action='store_true', default=False,
#                    help='Apply LR scheduler step on update instead of epoch end.')
# group.add_argument('--lr', type=float, default=None, metavar='LR',
#                    help='learning rate, overrides lr-base if set (default: None)')
# group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
#                    help='base learning rate: lr = lr_base * global_batch_size / base_size')
# group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
#                    help='base learning rate batch size (divisor, default: 256).')
# group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
#                    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
# group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
#                    help='learning rate noise on/off epoch percentages')
# group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
#                    help='learning rate noise limit percent (default: 0.67)')
# group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
#                    help='learning rate noise std-dev (default: 1.0)')
# group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
#                    help='learning rate cycle len multiplier (default: 1.0)')
# group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
#                    help='amount to decay each learning rate cycle (default: 0.5)')
# group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
#                    help='learning rate cycle limit, cycles enabled if > 1')
# group.add_argument('--lr-k-decay', type=float, default=1.0,
#                    help='learning rate k-decay for cosine/poly (default: 1.0)')
# group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
#                    help='warmup learning rate (default: 1e-5)')
# group.add_argument('--min-lr', type=float, default=0, metavar='LR',
#                    help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
# group.add_argument('--epochs', type=int, default=300, metavar='N',
#                    help='number of epochs to train (default: 300)')
# group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
#                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
# group.add_argument('--start-epoch', default=None, type=int, metavar='N',
#                    help='manual epoch number (useful on restarts)')
# group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
#                    help='list of decay epoch indices for multistep lr. must be increasing')
# group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
#                    help='epoch interval to decay LR')
# group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
#                    help='epochs to warmup LR, if scheduler supports')
# group.add_argument('--warmup-prefix', action='store_true', default=False,
#                    help='Exclude warmup period from decay schedule.'),
# group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
#                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
# group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
#                    help='patience epochs for Plateau LR scheduler (default: 10)')
# group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
#                    help='LR decay rate (default: 0.1)')

# # Augmentation & regularization parameters
# group = parser.add_argument_group('Augmentation and regularization parameters')
# group.add_argument('--no-aug', action='store_true', default=False,
#                    help='Disable all training augmentation, override other train aug args')
# group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
#                    help='Random resize scale (default: 0.08 1.0)')
# group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
#                    help='Random resize aspect ratio (default: 0.75 1.33)')
# group.add_argument('--hflip', type=float, default=0.5,
#                    help='Horizontal flip training aug probability')
# group.add_argument('--vflip', type=float, default=0.,
#                    help='Vertical flip training aug probability')
# group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
#                    help='Color jitter factor (default: 0.4)')
# group.add_argument('--aa', type=str, default=None, metavar='NAME',
#                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
# group.add_argument('--aug-repeats', type=float, default=0,
#                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
# group.add_argument('--aug-splits', type=int, default=0,
#                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
# group.add_argument('--jsd-loss', action='store_true', default=False,
#                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
# group.add_argument('--bce-loss', action='store_true', default=False,
#                    help='Enable BCE loss w/ Mixup/CutMix use.')
# group.add_argument('--bce-target-thresh', type=float, default=None,
#                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
# group.add_argument('--reprob', type=float, default=0., metavar='PCT',
#                    help='Random erase prob (default: 0.)')
# group.add_argument('--remode', type=str, default='pixel',
#                    help='Random erase mode (default: "pixel")')
# group.add_argument('--recount', type=int, default=1,
#                    help='Random erase count (default: 1)')
# group.add_argument('--resplit', action='store_true', default=False,
#                    help='Do not random erase first (clean) augmentation split')
# group.add_argument('--mixup', type=float, default=0.0,
#                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
# group.add_argument('--cutmix', type=float, default=0.0,
#                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
# group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
#                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
# group.add_argument('--mixup-prob', type=float, default=1.0,
#                    help='Probability of performing mixup or cutmix when either/both is enabled')
# group.add_argument('--mixup-switch-prob', type=float, default=0.5,
#                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
# group.add_argument('--mixup-mode', type=str, default='batch',
#                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
# group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
#                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
# group.add_argument('--smoothing', type=float, default=0.1,
#                    help='Label smoothing (default: 0.1)')
# group.add_argument('--train-interpolation', type=str, default='random',
#                    help='Training interpolation (random, bilinear, bicubic default: "random")')
# group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
#                    help='Dropout rate (default: 0.)')
# group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
#                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
# group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
#                    help='Drop path rate (default: None)')
# group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
#                    help='Drop block rate (default: None)')

# # Batch norm parameters (only works with gen_efficientnet based models currently)
# group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
# group.add_argument('--bn-momentum', type=float, default=None,
#                    help='BatchNorm momentum override (if not None)')
# group.add_argument('--bn-eps', type=float, default=None,
#                    help='BatchNorm epsilon override (if not None)')
# group.add_argument('--sync-bn', action='store_true',
#                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
# group.add_argument('--dist-bn', type=str, default='reduce',
#                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
# group.add_argument('--split-bn', action='store_true',
#                    help='Enable separate BN layers per augmentation split.')

# # Model Exponential Moving Average
# group = parser.add_argument_group('Model exponential moving average parameters')
# group.add_argument('--model-ema', action='store_true', default=False,
#                    help='Enable tracking moving average of model weights')
# group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
#                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
# group.add_argument('--model-ema-decay', type=float, default=0.9998,
#                    help='decay factor for model weights moving average (default: 0.9998)')

# # Misc
# group = parser.add_argument_group('Miscellaneous parameters')
# group.add_argument('--seed', type=int, default=42, metavar='S',
#                    help='random seed (default: 42)')
# group.add_argument('--worker-seeding', type=str, default='all',
#                    help='worker seed mode (default: all)')
# group.add_argument('--log_interval', type=int, default=50, metavar='N',
#                    help='how many batches to wait before logging training status')
# group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
#                    help='how many batches to wait before writing recovery checkpoint')
# group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
#                    help='number of checkpoints to keep (default: 10)')
# group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
#                    help='how many training processes to use (default: 4)')
# group.add_argument('--save-images', action='store_true', default=False,
#                    help='save images of input bathes every log interval for debugging')
# group.add_argument('--amp', action='store_true', default=False,
#                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
# group.add_argument('--amp-dtype', default='float16', type=str,
#                    help='lower precision AMP dtype (default: float16)')
# group.add_argument('--amp-impl', default='native', type=str,
#                    help='AMP impl to use, "native" or "apex" (default: native)')
# group.add_argument('--no-ddp-bb', action='store_true', default=False,
#                    help='Force broadcast buffers for native DDP to off.')
# group.add_argument('--pin-mem', action='store_true', default=False,
#                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
# group.add_argument('--no-prefetcher', action='store_true', default=False,
#                    help='disable fast prefetcher')
# group.add_argument('--output', default='', type=str, metavar='PATH',
#                    help='path to output folder (default: none, current dir)')
# group.add_argument('--experiment', default='', type=str, metavar='NAME',
#                    help='name of train experiment, name of sub-folder for output')
# group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
#                    help='Best metric (default: "top1"')
# group.add_argument('--tta', type=int, default=0, metavar='N',
#                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
# group.add_argument("--local_rank", default=0, type=int)
# group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
#                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
# group.add_argument('--log-wandb', action='store_true', default=False,
#                    help='log training and validation metrics to wandb')



def main():
    import config
    args = config
    
    utils.setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()
        
    trainer = Trainer(device=device, args=args)
    trainer.init()
    
    data_config = resolve_data_config(vars(args), model=trainer.model, verbose=utils.is_primary(args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        trainer.model, trainer.optimizer = amp.initialize(trainer.model, trainer.optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda':
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            trianer.model,
            args.resume,
            optimizer=None if args.no_resume_opt else trainer.optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            trainer.model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            trainer.model = ApexDDP(trainer.model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            trainer.model = NativeDDP(trainer.model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    dataset_train = build_train_dataset(args)
    dataset_eval = build_eval_dataset(args)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    if trainer.mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not trainer.num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if trainer.num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=trainer.num_aug_splits)

    loader_train = build_train_dataloader(dataset_train, data_config, trainer.num_aug_splits, collate_fn, device, args)
    loader_eval = build_eval_dataloader(dataset_eval, data_config, device, args)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=trainer.model,
            optimizer=trainer.optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler(
        trainer.optimizer,
        updates_per_epoch=updates_per_epoch,
        args=args,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
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

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = trainer.validate(
                trainer.model,
                loader_eval,
                trainer.validate_loss_fn,
                args,
                device=device,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    trainer.validate_loss_fn,
                    args,
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
                    log_wandb=args.log_wandb and has_wandb,
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

# coding=utf-8

import os
import sys

import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import torchvision

from dataloaders import transforms, presets
from dataloaders.sampler import RASampler
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch
from train import utils


def build_train_dataset(args):
    dist_pytorch.main_proc_print('building train dataset ...')
    traindir = os.path.join(args.data_dir, args.train_data)
    interpolation = InterpolationMode(args.interpolation)
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=args.train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
        ),
    )

    return train_dataset


def build_eval_dataset(args):
    dist_pytorch.main_proc_print('building eval dataset ...')
    valdir = os.path.join(args.data_dir, args.eval_data)
    interpolation = InterpolationMode(args.interpolation)
    preprocessing = presets.ClassificationPresetEval(
        crop_size=args.val_crop_size,
        resize_size=args.val_resize_size,
        interpolation=interpolation)

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )

    return val_dataset


def build_train_dataloader(train_dataset, args):
    """Training dataloaders."""
    dist_pytorch.main_proc_print('building train dataloaders ...')

    if dist_pytorch.is_dist_avail_and_initialized():
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dataset,
                                      shuffle=True,
                                      repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        dist_pytorch.main_proc_print(
            f"use sampler: DistributedSampler, num_replicas:{args.n_device}")
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    collate_fn = None
    num_classes = len(train_dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomCutmix(num_classes,
                                    p=1.0,
                                    alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    dist_pytorch.main_proc_print(
        f'train samples:{len(train_dataset)}, batch size:{args.train_batch_size}'
    )
    return train_dataloader


def build_eval_dataloader(eval_dataset, args):
    """Training and validation dataloaders."""
    dist_pytorch.main_proc_print('building eval dataloaders ...')

    if dist_pytorch.is_dist_avail_and_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, drop_last=True)
        dist_pytorch.main_proc_print(
            f"use sampler: DistributedSampler, num_replicas:{args.n_device}")
    else:
        val_sampler = torch.utils.data.SequentialSampler(eval_dataset)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True)

    dist_pytorch.main_proc_print(
        f'eval samples:{len(eval_dataset)}, batch size:{args.eval_batch_size}')
    return eval_dataloader

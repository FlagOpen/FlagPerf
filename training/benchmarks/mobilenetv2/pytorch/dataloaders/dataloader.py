# coding=utf-8

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch


def build_train_dataset(args):
    traindir = os.path.join(args.data_dir, args.train_data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset


def build_eval_dataset(args):
    valdir = os.path.join(args.data_dir, args.eval_data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return val_dataset


def build_train_dataloader(train_dataset, args):
    """Traing dataloaders."""
    dist_pytorch.main_proc_print('building train dataloaders ...')

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        dist_pytorch.main_proc_print(
            f"use sampler: DistributedSampler, num_replicas:{args.n_device}")
    else:
        train_sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    dist_pytorch.main_proc_print(
        f'train samples:{len(train_dataset)}, batch size:{args.train_batch_size}'
    )
    return train_dataloader


def build_eval_dataloader(eval_dataset, args):
    """Traing and validation dataloaders."""
    dist_pytorch.main_proc_print('building eval dataloaders ...')

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, drop_last=True)
        dist_pytorch.main_proc_print(
            f"use sampler: DistributedSampler, num_replicas:{args.n_device}")
    else:
        val_sampler = None

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler)

    dist_pytorch.main_proc_print(
        f'eval samples:{len(eval_dataset)}, batch size:{args.eval_batch_size}')
    return eval_dataloader

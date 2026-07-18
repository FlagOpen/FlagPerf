# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

from os.path import join as pjoin
import os
import sys
import random
import numpy as np
import torch
import torchvision as tv
import torch.distributed as dist

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch


def build_train_dataset(args):
    precrop, crop = (512, 480)
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.RandomCrop((crop, crop)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = tv.datasets.ImageFolder(pjoin(args.data_dir, args.train_data),
                                        train_tx)
    return train_set


def build_eval_dataset(args):
    precrop, crop = (512, 480)
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    valid_set = tv.datasets.ImageFolder(pjoin(args.data_dir, args.eval_data),
                                        val_tx)
    return valid_set


def build_train_dataloader(train_dataset, args):
    """Training dataloaders."""
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False)

    dist_pytorch.main_proc_print(
        f'train samples:{len(train_dataset)}, batch size:{args.batch_size}')
    return train_dataloader


def build_eval_dataloader(eval_dataset, args):
    """Validation dataloaders."""
    dist_pytorch.main_proc_print('building eval dataloaders ...')

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, drop_last=True)
        dist_pytorch.main_proc_print(
            f"use sampler: DistributedSampler, num_replicas:{args.n_device}")
    else:
        val_sampler = None

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  sampler=val_sampler,
                                                  drop_last=False)

    dist_pytorch.main_proc_print(
        f'eval samples:{len(eval_dataset)}, batch size:{args.batch_size}')
    return eval_dataloader

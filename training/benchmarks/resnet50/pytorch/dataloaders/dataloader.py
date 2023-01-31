# coding=utf-8

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch


class WorkerInitializer(object):

    _instance = None

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, idx):
        np.random.seed(seed=self.seed + idx)
        random.seed(self.seed + idx)

    @classmethod
    def default(cls, seed=0):
        if cls._instance is None:
            cls._instance = cls(seed)
        return cls._instance


def build_data_loader(dataset,
                      batch_size,
                      num_workers,
                      drop_last,
                      shuffle=True,
                      only_rank0=False,
                      worker_init_fn: WorkerInitializer = None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""
    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default()
    world_size = dist.get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, shuffle=shuffle)
    dist_pytorch.main_proc_print( f"use sampler: DistributedSampler, num_replicas:{world_size}")

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)
    return data_loader


def build_train_dataloader(args, train_dataset, worker_init_fn: WorkerInitializer = None):
    """Traing and validation dataloaders."""
    dist_pytorch.main_proc_print('building train dataloaders ...')
    train_dataloader = build_data_loader(train_dataset,
                                         args.train_batch_size,
                                         args.num_workers,
                                         drop_last=False,
                                         worker_init_fn=worker_init_fn)

    dist_pytorch.main_proc_print(
        f'train samples:{len(train_dataset)}, batch size:{args.train_batch_size}'
    )
    return train_dataloader


def build_eval_dataloaders(args, eval_dataset):
    dist_pytorch.main_proc_print('building eval dataloaders ...')
    eval_dataloader = build_data_loader(eval_dataset,
                                        args.eval_batch_size,
                                        args.num_workers,
                                        shuffle=False,
                                        drop_last=False)
    dist_pytorch.main_proc_print( f'eval samples:{len(eval_dataset)}, batch size:{args.eval_batch_size}')
    return eval_dataloader

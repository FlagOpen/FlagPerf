# coding=utf-8
import os
import sys
import torch
from .transforms import Compose, ToTensor, RandomHorizontalFlip

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch

from .dataset_coco import CocoDetection

from utils.train import create_aspect_ratio_groups, GroupedBatchSampler

data_transform = {
    "train": Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
    ]),
    "val": Compose([ToTensor()])
}


def build_train_dataset(config):
    COCO_root = config.data_dir
    train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    return train_dataset


def build_eval_dataset(config):
    COCO_root = config.data_dir
    val_dataset = CocoDetection(COCO_root, "val", data_transform["val"])
    return val_dataset


def build_train_dataloader(
    config,
    train_dataset,
):
    dist_pytorch.main_proc_print("building train dataloader ...")

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    if config.aspect_ratio_group_factor >= 0:
        # 统计所有图像比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(
            train_dataset, k=config.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids,
                                                  config.train_batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, config.train_batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.num_workers,
        collate_fn=train_dataset.collate_fn)
    return data_loader, train_sampler


def build_eval_dataloader(config, train_dataset, eval_dataset):
    dist_pytorch.main_proc_print("building eval dataloaders ...")
    rank = dist_pytorch.get_rank()

    eval_sampler = None

    if config.distributed:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, drop_last=True, rank=rank)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        sampler=eval_sampler,
        num_workers=config.num_workers,
        collate_fn=train_dataset.collate_fn)

    dist_pytorch.main_proc_print(
        f"eval samples:{len(eval_dataset)}, batch size:{config.eval_batch_size}"
    )
    return eval_dataloader

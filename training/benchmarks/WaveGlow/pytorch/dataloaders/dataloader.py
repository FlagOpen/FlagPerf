import copy
import os

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from . import data_function
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from driver import dist_pytorch


def get_collate_fn(args):
    collate_fn = data_function.get_collate_function(args.name)
    return collate_fn


def build_train_dataset(args):
    trainset = data_function.get_data_loader(args.name, args.data_dir,
                                              args.training_files, args)
    return trainset


def build_train_dataloader(trainset, args):
    if dist_pytorch.get_world_size() > 1:
        train_sampler = DistributedSampler(trainset, seed=(args.seed or 0))
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(trainset,
                              num_workers=1,
                              shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=get_collate_fn(args))
    return train_loader


def build_eval_dataloader(valset, args):
    val_sampler = DistributedSampler(
        valset) if dist_pytorch.get_world_size() > 1 else None
    val_loader = DataLoader(
        valset,
        num_workers=1,
        shuffle=False,
        sampler=val_sampler,
        batch_size=args.batch_size,
        pin_memory=False,
        collate_fn=get_collate_fn(args),
        drop_last=(True if args.bench_class == "perf-train" else False))

    return val_loader


def build_eval_dataset(args):
    valset = data_function.get_data_loader(args.name, args.data_dir,
                                            args.validation_files, args)
    return valset


class FilterAndRemapCocoCategories(object):

    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

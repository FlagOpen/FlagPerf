# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    # config.defrost()
    dataset_train, config.model_num_classes = build_dataset(is_train=True, config=config)
    # config.freeze()
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.data_zip_mode and config.data_cache_mode == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.test_sequential:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.test_shuffle
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.train_batch_size,
        num_workers=config.data_num_workers,
        pin_memory=config.data_pin_memory,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.data_num_workers,
        pin_memory=config.data_pin_memory,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.aug_mixup > 0 or config.aug_cutmix > 0. or config.aug_cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.aug_mixup, cutmix_alpha=config.aug_cutmix, cutmix_minmax=config.aug_cutmix_minmax,
            prob=config.aug_mixup_prob, switch_prob=config.aug_mixup_switch_prob, mode=config.aug_mixup_mode,
            label_smoothing=config.model_label_smoothing, num_classes=config.model_num_classes)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.data_dataset == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.data_zip_mode:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.train_data_path, ann_file, prefix, transform,
                                        cache_mode=config.data_cache_mode if is_train else 'part')
        else:
            root = os.path.join(config.train_data_path, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.data_img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.data_img_size,
            is_training=True,
            color_jitter=config.aug_color_jitter if config.aug_color_jitter > 0 else None,
            auto_augment=config.aug_auto_augment if config.aug_auto_augment != 'none' else None,
            re_prob=config.aug_reprob,
            re_mode=config.aug_remode,
            re_count=config.aug_recount,
            interpolation=config.data_interpolation,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.data_img_size, padding=4)
        return transform

    t = []
    if resize_im:
        if config.test_crop:
            size = int((256 / 224) * config.data_img_size)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.data_interpolation)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.data_img_size))
        else:
            t.append(
                transforms.Resize((config.data_img_size, config.data_img_size),
                                  interpolation=_pil_interp(config.data_interpolation))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

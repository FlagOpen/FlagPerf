import os
import functools

import torch
from torch.utils.data.distributed import DistributedSampler

from data.data_loader import NumpyTupleDataset
from data import transform
from misc.config import CONFIGS


def build_datasets(args):

    # Model configuration
    assert args.config_name in CONFIGS
    config = CONFIGS[args.config_name]
    data_file = config.dataset_config.dataset_file
    transform_fn = functools.partial(transform.transform_fn, config=config)
    valid_idx = transform.get_val_ids(config, args.data_dir)

    # Datasets:
    data_file_path = os.path.join(args.data_dir, data_file)
    print(f"data_file_path: {data_file_path}")

    dataset = NumpyTupleDataset.load(
        os.path.join(args.data_dir, data_file),
        transform=transform_fn,
    )
    if len(valid_idx) == 0:
        raise ValueError('Empty validation set!')
    else:
        print(f"valid_idx size: {len(valid_idx)}")

    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, valid_idx)
    return train_dataset, test_dataset



def _get_sampler(args, train_dataset):
    if args.distributed:
        sampler = DistributedSampler(train_dataset,
                                     seed=args.seed,
                                     drop_last=False)
    else:
        sampler = None
    return sampler


def build_train_dataloader(args, train_dataset):
    sampler = _get_sampler(args, train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    if args.distributed:
        train_dataloader.sampler.set_epoch(-1)
    return train_dataloader


def build_eval_dataloader(args, eval_dataset):
    return None

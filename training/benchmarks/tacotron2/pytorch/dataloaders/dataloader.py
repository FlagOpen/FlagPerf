from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data_functions import get_data_loader, get_collate_function


def build_train_dataset(args):
    train_dataset = get_data_loader(args.data_dir, args.training_files, args)
    return train_dataset


def build_train_dataloader(args,
                           train_dataset,
                           n_frames_per_step: int = 1,
                           distributed_run: bool = True):

    collate_fn = get_collate_function(n_frames_per_step)
    if distributed_run:
        train_sampler = DistributedSampler(train_dataset,
                                           seed=(args.seed or 0))
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = DataLoader(train_dataset,
                                  num_workers=1,
                                  shuffle=shuffle,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  pin_memory=False,
                                  drop_last=True,
                                  collate_fn=collate_fn)
    return train_dataloader


def build_eval_dataset(args):
    validate_set = get_data_loader(args.data_dir, args.validation_files, args)
    return validate_set


def build_eval_dataloader(val_dataset, args):
    val_sampler = DistributedSampler(val_dataset) if args.distributed else None
    collate_fn = get_collate_function()
    val_dataloader = DataLoader(val_dataset,
                                num_workers=1,
                                shuffle=False,
                                sampler=val_sampler,
                                batch_size=args.eval_batch_size,
                                pin_memory=False,
                                collate_fn=collate_fn,
                                drop_last=False)
    return val_dataloader
import os
import sys
import torch
import time

from contextlib import contextmanager

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch

# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license


@contextmanager
def evaluating(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


class Evaluator:

    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader

    def evaluate(self, trainer):
        """Handles all the validation scoring and printing"""
        model = trainer.model
        criterion = trainer.criterion
        validate_dataset = trainer.validate_dataset

        # TODO

        batch_iter = None
        world_size = None
        collate_fn = None
        batch_to_gpu = None
        amp_run = trainer.config.amp

        distributed_run = trainer.config.distributed
        epoch = trainer.epoch
        batch_size = trainer.eval_batch_size

        with evaluating(model), torch.no_grad():
            val_sampler = None
            if distributed_run:
                val_sampler = DistributedSampler(validate_dataset)
                

            val_loader = DataLoader(validate_dataset,
                                    num_workers=1,
                                    shuffle=False,
                                    sampler=val_sampler,
                                    batch_size=batch_size,
                                    pin_memory=False,
                                    collate_fn=collate_fn,
                                    drop_last=False)

            val_loss = 0.0
            num_iters = 0
            val_items_per_sec = 0.0

            for i, batch in enumerate(val_loader):
                torch.cuda.synchronize()
                iter_start_time = time.perf_counter()

                x, y, num_items = batch_to_gpu(batch)
                # AMP upstream autocast
                with torch.cuda.amp.autocast(enabled=amp_run):
                    y_pred = model(x)
                    loss = criterion(y_pred, y)

                if distributed_run:
                    reduced_val_loss = reduce_tensor(loss.data,
                                                     world_size).item()
                    reduced_num_items = reduce_tensor(num_items.data, 1).item()
                else:  #
                    reduced_val_loss = loss.item()
                    reduced_num_items = num_items.item()
                val_loss += reduced_val_loss

                torch.cuda.synchronize()
                iter_stop_time = time.perf_counter()
                iter_time = iter_stop_time - iter_start_time

                items_per_sec = reduced_num_items / iter_time
                val_items_per_sec += items_per_sec
                num_iters += 1

            val_loss = val_loss / num_iters
            val_items_per_sec = val_items_per_sec / num_iters

            return val_loss, val_items_per_sec


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / num_gpus
    else:
        rt = torch.div(rt, num_gpus, rounding_mode='floor')
    return rt

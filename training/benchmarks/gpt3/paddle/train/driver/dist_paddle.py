import paddle
import paddle.distributed as dist
import logging
import random
import os
import numpy as np
from contextlib import contextmanager
from icecream import ic

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    paddle.seed(config.seed)

def barrier():
    if dist.is_initialized():
        dist.barrier()


def get_rank(default=0):
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = default
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    return world_size


def main_proc_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def init_dist_training_env(config):
    paddle.device.set_device("gpu")
    if dist.get_world_size() <= 1:
        config.device = paddle.device.get_device()
        config.world_size = get_world_size()
    else:
        dist.init_parallel_env()
        config.device = paddle.device.get_device()
        config.world_size = get_world_size()
        print('------------------------')
        print('device numbers:', config.world_size)
        print('the processing uses', config.device)
        return


def global_batch_size(config):
    return config.per_device_train_batch_size * config.world_size


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()


def is_main_process():
    if dist.is_initialized():
        if "PADDLE_TRAINER_ID" in os.environ:
            return int(os.environ["PADDLE_TRAINER_ID"]) == 0
        else:
            return dist.get_rank() == 0

    return True


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s

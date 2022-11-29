import paddle
import paddle.distributed as dist
import logging
import random
import os
from contextlib import contextmanager

# from torch.distributed import (
#     is_available,
#     is_initialized,
#     init_process_group,
#     broadcast,
#     all_reduce
# )


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).

    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    # if is_available() and is_initialized():
    #     seeds_tensor = torch.LongTensor(seeds).to(device)
    #     broadcast(seeds_tensor, 0)
    #     seeds = seeds_tensor.tolist()
    # return seeds
    if dist.is_initialized():
        #seeds_tensor = paddle.to_tensor(seeds,dtype='float64').to(device)
        #seeds_tensor = paddle.to_tensor(seeds,dtype='float64',place=paddle.CUDAPlace())
        seeds_tensor = paddle.to_tensor(seeds, dtype='int64')
        dist.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def setup_seeds(master_seed, epochs, device):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            logging.info(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        logging.info(f'Using master seed from command line: {master_seed}')

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers

    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)

    return worker_seeds, shuffling_seeds


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if dist.is_initialized():
        dist.all_reduce(paddle.to_tensor(1))
        paddle.device.cuda.synchronize()


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
    if dist.get_world_size() <= 1:
        #   paddle.set_device('gpu')
        config.device = paddle.device.get_device()
        config.n_device = get_world_size()
    else:
        #paddle.set_device('gpu')
        dist.init_parallel_env()
        config.device = paddle.device.get_device()
        config.n_device = get_world_size()
        print('------------------------')
        print('device numbers:', config.n_device)
        print('the processing uses', config.device)
        return


def global_batch_size(config):

    return config.train_batch_size * config.n_device


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

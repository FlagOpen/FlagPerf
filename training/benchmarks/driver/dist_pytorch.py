# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# Modified some functions to support FlagPerf.
#
# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
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
import logging
import random
import os
from contextlib import contextmanager

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
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


def get_rank(default=0):
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = default
    return rank


def get_world_size(vendor="nvidia"):
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if vendor == "kunlun":
        # TODO
        pass
    else:  # nvidia
        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return world_size


def main_proc_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def set_device(cuda, local_rank):
    """
    TODO: Support other accelarators.
    Sets device based on local_rank and returns instance of torch.device.

    :param cuda: if True: use cuda
    :param local_rank: local rank of the worker
    """
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def barrier(vendor="nvidia"):
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if vendor == "kunlun":
            # TODO
            pass
        else:
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
            torch.cuda.synchronize()


def init_dist_training_env(config):
    ''' TODO: Support other accelarators.  '''
    if config.vendor == "kunlun":
        # TODO
        pass
    else:  # nvidia
        if int(os.environ.get("WORLD_SIZE", 1)) <= 1:
            config.device = torch.device("cuda")
            config.n_device = 1
        else:
            torch.cuda.set_device(config.local_rank)
            host_addr_full = 'tcp://' + os.environ[
                "MASTER_ADDR"] + ':' + os.environ["MASTER_PORT"]
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.distributed.init_process_group(backend=config.dist_backend,
                                                 init_method=host_addr_full,
                                                 rank=rank,
                                                 world_size=world_size)
            config.device = torch.device("cuda", config.local_rank)
            config.n_device = torch.distributed.get_world_size()


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
    if torch.distributed.is_initialized():
        if "LOCAL_RANK" in os.environ:
            return int(os.environ["LOCAL_RANK"]) == 0
        else:
            return get_rank() == 0

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


class PyTorchDistributedDataParallel(DDP):

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
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

from typing import TypeVar, List

import torch
import numpy as np
from torch.utils.data import (RandomSampler, Sampler, DistributedSampler as
                              TorchDistributedSampler)

from common.fairseq.data import data_utils

T = TypeVar('T')


class DistributedSampler(Sampler):

    def __init__(self, dataset, batch_size, world_size, rank):
        """
        Constructor for the DistributedSampler.
        :param dataset: dataset
        :param batch_size: local batch size
        :param world_size: number of distributed workers
        :param rank: rank of the current process
        """
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0

        self.batch_size = batch_size
        self.global_batch_size = batch_size * world_size

        self.data_len = len(self.dataset)

        self.num_samples = self.data_len // self.global_batch_size \
            * self.global_batch_size

    def distribute_batches(self, indices):
        """
        Assigns batches to workers.
        Consecutive ranks are getting consecutive batches.
        :param indices: torch.tensor with batch indices
        """
        assert len(indices) == self.num_samples

        indices = indices.view(-1, self.batch_size)
        indices = indices[self.rank::self.world_size].contiguous()
        indices = indices.view(-1)
        indices = indices.tolist()

        assert len(indices) == self.num_samples // self.world_size
        return indices

    def reshuffle_batches(self, indices, rng):
        """
        Permutes global batches
        :param indices: torch.tensor with batch indices
        :param rng: instance of torch.Generator
        """
        indices = indices.view(-1, self.global_batch_size)
        num_batches = indices.shape[0]
        order = torch.randperm(num_batches, generator=rng)
        indices = indices[order, :]
        indices = indices.view(-1)
        return indices

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # generate permutation
        indices = torch.randperm(self.data_len, generator=g)

        # make indices evenly divisible by (batch_size * world_size)
        indices = indices[:self.num_samples]

        # assign batches to workers
        indices = self.distribute_batches(indices)
        return iter(indices)

    def set_epoch(self, epoch):
        """
        Sets current epoch index.
        Epoch index is used to seed RNG in __iter__() function.
        :param epoch: index of current epoch
        """
        self.epoch = epoch

    def __len__(self):
        return self.num_samples // self.world_size


class DistributedIndicesSampler(TorchDistributedSampler):
    """ DistributedSampler operating on indices.

    Differences wrt. DistributedSampler:
    1) use Numpy RNG instead of PyTorch RNG
    2) treat `self.dataset` as indices - DistributedSampler assumes indices
        are determined with `range(len(self.dataset))`
    3) if `drop_last` is False, pad indices with `fillvalue`
        or don't pad at all if `fillvalue` is None (useful for validation)
    """

    def __init__(self, *args, fillvalue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fillvalue = fillvalue
        if not self.drop_last and self.fillvalue is None:
            self.total_size = len(self.dataset)
            # possibly different num_samples for each device,
            # this will work with DDP only for validation
            self.num_samples = len(
                range(self.rank, self.total_size, self.num_replicas))

    def __iter__(self):
        indices = list(self.dataset)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            with data_utils.numpy_seed(self.seed + self.epoch):
                np.random.shuffle(indices)

        if not self.drop_last:
            if self.fillvalue is not None:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                indices += [self.fillvalue] * padding_size
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

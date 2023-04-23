import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Union, Tuple, Any, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import utils
from .dataset import PretrainingDataset


def get_sampler(dataset, sampler_type):
    return dict(random=RandomSampler,
                sequential=SequentialSampler,
                distributed=DistributedSampler)[sampler_type.lower()](dataset)


class WorkerInitializer(object):

    _instance = None

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, idx):
        np.random.seed(seed=self.seed + idx)
        random.seed(self.seed + idx)

    @classmethod
    def default(cls, seed=0):
        if cls._instance is None:
            cls._instance = cls(seed)
        return cls._instance


# sampler: Random | Sequential | Distributed
def create_train_dataloader(dataset,
                            batch_size,
                            worker_init_fn: WorkerInitializer = None,
                            sampler_type='Random',
                            pin_memory=True):
    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default()
    sampler = get_sampler(dataset, sampler_type)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=0 if batch_size <= 8 else 8,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
    )

    return dataloader


def create_eval_dataloader(eval_dir, eval_batch_size, max_predictions_per_seq,
                           num_eval_examples, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(eval_dir)):
        eval_file_path = os.path.join(eval_dir, eval_file)
        if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
            eval_data.extend(
                PretrainingDataset(eval_file_path,
                                   max_pred_length=max_predictions_per_seq))
            if len(eval_data) > num_eval_examples:
                eval_data = eval_data[:num_eval_examples]
                break

    if torch.distributed.is_initialized():
        chunk_size = num_eval_examples // torch.distributed.get_world_size()
        eval_sampler = DistributedSampler(eval_data, shuffle=False)
    else:
        chunk_size = num_eval_examples
        eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        num_workers=0 if min(chunk_size, eval_batch_size) <= 10 else 4,
        worker_init_fn=worker_init_fn,
        pin_memory=True)
    return eval_dataloader


class PretrainingDataloaders:

    def __init__(self,
                 train_dir: str,
                 max_predictions_per_seq: int,
                 batch_size: int = 2,
                 shuffle: bool = True,
                 seed: Union[int, list] = 0,
                 num_replicas: int = None,
                 rank: int = None,
                 num_files_per_iter: int = 1,
                 worker_init: WorkerInitializer = None,
                 pin_memory: bool = True,
                 pool: ProcessPoolExecutor = None):
        self.train_dir = train_dir
        self.max_predictions_per_seq = max_predictions_per_seq
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_files_per_iter = num_files_per_iter
        self.worker_init = worker_init
        self.pin_memory = pin_memory

        self.files = self.get_files()
        self.num_files = len(self.files)

        if num_replicas is None:
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        self.num_replicas = num_replicas

        if rank is None:
            rank = utils.get_rank()
        self.rank = rank

        self.num_files_per_replica = int(
            math.ceil(self.num_files / self.num_replicas))
        self.total_files = self.num_files_per_replica * self.num_replicas

        self.files_per_replica: List[str] = None

        # Prefetch dataloader
        self.pool = pool
        self.prefetched_dataloader_future: Future = None

        self.sub_dataloader: DataLoader = None

    def get_seed(self, epoch=0):
        if isinstance(self.seed, (tuple, list)):
            return self.seed[epoch]

        return self.seed + epoch

    def get_files(self):
        join = os.path.join
        files = [
            join(self.train_dir, f) for f in os.listdir(self.train_dir)
            if os.path.isfile(join(self.train_dir, f)) and 'part' in f
        ]

        files.sort()

        return files

    def set_epoch(self, epoch):
        if self.shuffle:
            random.Random(self.get_seed(epoch)).shuffle(self.files)

        files_per_replica = self.files[self.rank:self.total_files:self.
                                       num_replicas]
        padding_size = self.num_files_per_replica - len(files_per_replica)
        if padding_size > 0:
            files_per_replica = files_per_replica + self.files[:padding_size]
        self.files_per_replica = files_per_replica

    @staticmethod
    def next_dataloader(idx: int, max_predictions_per_seq: int,
                        files_per_replica: List, num_files_per_iter: int,
                        batch_size: int, shuffle: bool,
                        worker_init: WorkerInitializer, pin_memory: bool):
        files_per_iter = files_per_replica[idx * num_files_per_iter:(idx + 1) *
                                           num_files_per_iter]
        datasets = []
        for file in files_per_iter:
            datasets.append(PretrainingDataset(file, max_predictions_per_seq))

        datasets = ConcatDataset(datasets)
        sampler_type = "Random" if shuffle else "Sequential"
        return create_train_dataloader(datasets,
                                       batch_size,
                                       worker_init,
                                       sampler_type=sampler_type,
                                       pin_memory=pin_memory)

    def iter_batchs(self) -> Tuple[int, int, Any]:
        for dataloader_idx, sub_dataloader in enumerate(self):
            for batch_idx, batch in enumerate(sub_dataloader):
                yield dataloader_idx, batch_idx, batch

    def __iter__(self):
        self._next_index = 0
        self._num_iters = int(
            math.ceil(self.num_files_per_replica / self.num_files_per_iter))
        return self

    def __next__(self) -> DataLoader:
        if self._next_index < self._num_iters:
            next_dataloader_args = dict(
                max_predictions_per_seq=self.max_predictions_per_seq,
                files_per_replica=self.files_per_replica,
                num_files_per_iter=self.num_files_per_iter,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                worker_init=self.worker_init,
                pin_memory=self.pin_memory)
            self.prefetch_dataloader(self._next_index + 1,
                                     **next_dataloader_args)
            if self._next_index == 0 or self.pool is None:
                data = self.next_dataloader(idx=self._next_index,
                                            **next_dataloader_args)
            else:
                data = self.prefetched_dataloader_future.result()
            self._next_index += 1
            self.sub_dataloader = data
            return data
        else:
            raise StopIteration()

    def prefetch_dataloader(self, idx, *args, **kwargs):
        if self.pool is not None:
            self.prefetched_dataloader_future = self.pool.submit(
                self.next_dataloader, idx=idx, *args, **kwargs)

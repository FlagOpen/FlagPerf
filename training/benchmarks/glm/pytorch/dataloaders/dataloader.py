# coding=utf-8
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
import h5sparse
from scipy.sparse import csr_matrix
import glob
import os
import utils


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


class H5pyDataSet(Dataset):
    def __init__(self, split, args):
        self.split = split
        self.args = args
        self.load()

    def load(self):
        # used h5py load data
        if self.split == 'train':
            h5_file = self.args.train_data
        elif self.split == 'eval':
            h5_file = self.args.eval_data
        else:
            assert "split should be 'train' or 'evel'"

        self.data = {'text': [], 'answer_idx': [], 'position': [],
                     'mask': [], 'target': [], 'logit_mask': [],
                     'choice_start_end': [], 'answer_start_end': []}

        with h5sparse.File(h5_file, 'r') as f:
            for key in f.keys():
                # use sparse matrix
                if key in ['text', 'position', 'target', 'logit_mask']:
                    self.data[key] = csr_matrix(f[key][()])
                else:
                    if key in ['answer_start_end', 'answer_idx'] and self.split == 'train':
                        continue
                    self.data[key] = f[key][()].toarray()

        self.data['answer_idx'] = np.array(
            self.data['answer_idx']).reshape([-1])
        self.data['mask'] = np.array(self.data['mask']).reshape([-1])

        # # for key in self.data:
        # #     print(key, self.data[key].shape)
        # #     print(self.data['choice_start_end'][-1])

    def __len__(self):
        return len(self.data['choice_start_end'])

    def __getitem__(self, idx):
        sample = {'text': [], 'position': [],
                  'mask': [], 'target': [],
                  'logit_mask': []}
        c_start, c_end = self.data['choice_start_end'][idx]
        sample['mask'] = self.data['mask'][c_start:c_end]
        for i in range(c_start, c_end):
            # print(self.data['text'].getrow(i).toarray())
            sample['text'].append(self.data['text'].getrow(i).toarray())
            sample['target'].append(self.data['target'].getrow(i).toarray())
            sample['logit_mask'].append(
                self.data['logit_mask'].getrow(i).toarray())

            sample['position'].append(
                self.data['position'].getrow(2*i).toarray())
            sample['position'].append(
                self.data['position'].getrow(2*i+1).toarray())
        sample['text'] = np.concatenate(sample['text'], axis=0)
        sample['position'] = np.concatenate(sample['position'], axis=0)
        sample['position'] = sample['position'].reshape(
            [-1, 2, sample['position'].shape[-1]])
        sample['target'] = np.concatenate(sample['target'], axis=0)
        sample['logit_mask'] = np.concatenate(sample['logit_mask'], axis=0)

        if self.split == 'eval':
            a_start, a_end = self.data['answer_start_end'][idx]
            sample['answer_idx'] = self.data['answer_idx'][a_start:a_end]

        # for key in sample:
        #     print(key,sample[key].shape)
        return sample


def my_collate(batch):
    # if train text position mask target logit_mask
    # if eval text position mask target logit_mask answer_idx

    choice_nums = 0
    for sample in batch:
        choice_nums = max(choice_nums, len(sample['text']))

    def pad_choice_dim(data, choice_num):
        if len(data) < choice_num:
            data = np.concatenate([data] + [data[0:1]]
                                  * (choice_num - len(data)))
        return data

    new_batch = []
    answers = []
    for i, sample in enumerate(batch):
        new_sample = {}
        text_len = len(sample['text'])
        loss_mask = np.array([1] * text_len + [0] * (choice_nums - text_len),
                             dtype=np.int64)
        new_sample['loss_mask'] = loss_mask
        new_sample['label'] = 0
        for key, value in sample.items():
            if key != "answer_idx":
                new_sample[key] = pad_choice_dim(value, choice_nums)
            else:
                answers.append(sample['answer_idx'])

        new_batch.append(new_sample)

    new_batch = default_collate(new_batch)
    if len(answers):
        new_batch['answer_idx'] = answers

    return new_batch


def build_data_loader(dataset, batch_size, num_workers, drop_last, shuffle=True, only_rank0=False, worker_init_fn: WorkerInitializer = None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""
    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default()
    world_size = dist.get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, shuffle=shuffle)
    utils.main_proc_print(
        f"use sampler: DistributedSampler, num_replicas:{world_size}")

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=my_collate,
                                              worker_init_fn=worker_init_fn)
    return data_loader


def build_train_dataloader(args, worker_init_fn: WorkerInitializer = None):
    """Traing and validation dataloaders."""
    utils.main_proc_print('building train dataloaders ...')
    train_dataset = H5pyDataSet("train", args)
    train_dataloader = build_data_loader(
        train_dataset, args.train_batch_size, args.num_workers, drop_last=False, worker_init_fn=worker_init_fn)

    utils.main_proc_print(
        f'train samples:{len(train_dataset)}, batch size:{args.train_batch_size}')
    return train_dataloader


def build_eval_dataloaders(args):
    utils.main_proc_print('building eval dataloaders ...')
    eval_dataset = H5pyDataSet("eval", args)
    eval_dataloader = build_data_loader(
        eval_dataset, args.eval_batch_size, args.num_workers, shuffle=False, drop_last=False)
    utils.main_proc_print(
        f'eval samples:{len(eval_dataset)}, batch size:{args.eval_batch_size}')
    return eval_dataloader


if __name__ == "__main__":
    import config
    config.eval_data = "/mnt/dataset/mlperf/glm/ReCoRD/eval_hdf5/eval_sparse.hdf5"
    dataset = H5pyDataSet('eval', config)
    print("len:", len(dataset))
    sample = dataset[9000]

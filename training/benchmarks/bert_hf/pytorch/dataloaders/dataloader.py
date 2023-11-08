from torch.utils.data import DataLoader, Dataset
import torch
import random
import os
import numpy as np

class BertPretrainDataset(Dataset):

    def __init__(self, npy_file, item_length, mask_ratio):
        data = np.load(npy_file)
        mask_input = np.copy(data)
        
        num_mask = int(len(mask_input) * mask_ratio)
        mask_indices = np.random.choice(len(mask_input), num_mask, replace=False)
        mask_input[mask_indices] = 103
        
        self.data = torch.from_numpy(data)
        self.input_ids = torch.from_numpy(mask_input)
        self.item_length = item_length
        self.length = len(data) // item_length * item_length

    def __getitem__(self, index):
        start = index * self.item_length
        end = start + self.item_length
        label_ids = self.data[start:end]
        input_ids = self.input_ids[start:end]
        
        return input_ids, label_ids

    def __len__(self):
        return self.length // self.item_length


def get_bert_dataset(config):
    trainset = BertPretrainDataset(os.path.join(config.data_dir, config.datafilename),
                                    config.seq_length, config.mask_ratio)
    valset = BertPretrainDataset(os.path.join(config.data_dir, config.valdatafilename),
                                    config.seq_length, config.mask_ratio)
    return trainset, valset


def build_dataloader(config):
    trainset, valset = get_bert_dataset(config)

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    else:
        train_sampler = torch.utils.data.RandomSampler(trainset)
        val_sampler = torch.utils.data.RandomSampler(valset)

    trainloader = DataLoader(trainset,
                             batch_size=config.train_batch_size,
                             sampler=train_sampler,
                             num_workers=config.num_workers,
                             drop_last=True,
                             pin_memory=True)
    valloader = DataLoader(valset,
                           batch_size=config.eval_batch_size,
                           sampler=val_sampler,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True)

    return trainloader, valloader

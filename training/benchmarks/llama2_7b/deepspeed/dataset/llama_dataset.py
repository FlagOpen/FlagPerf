import numpy as np
import torch
import os
from torch.utils.data import Dataset


class Llama2PretrainDataset(Dataset):

    def __init__(self, npy_file, item_length):
        data = np.load(npy_file)
        self.data = torch.from_numpy(data)
        self.item_length = item_length
        self.length = len(data) // item_length * item_length

    def __getitem__(self, index):
        start = index * self.item_length
        end = start + self.item_length
        return self.data[start:end]

    def __len__(self):
        return self.length // self.item_length


def get_llama_dataset(args, seqlength, datafilename):
    dataset = Llama2PretrainDataset(os.path.join(args.data_dir, datafilename),
                                    seqlength)
    return dataset

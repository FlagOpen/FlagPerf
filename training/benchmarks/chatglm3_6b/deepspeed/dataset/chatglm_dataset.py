# Copyright 2026 FlagOS Contributors
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

import numpy as np
import torch
import os
from torch.utils.data import Dataset


class Chatglm3PretrainDataset(Dataset):

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


def get_chatglm_dataset(args, seqlength, datafilename):
    dataset = Chatglm3PretrainDataset(os.path.join(args.data_dir, datafilename),
                                    seqlength)
    return dataset

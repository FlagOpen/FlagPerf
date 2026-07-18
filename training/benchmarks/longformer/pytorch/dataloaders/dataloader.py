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

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datasets
from transformers import LongformerTokenizer, DataCollatorForLanguageModeling

class LongformerDataset(Dataset):
    def __init__(self, filepath):
        origin_data = np.load(filepath)
        self.input_ids = origin_data['input_ids']
        self.special_tokens_mask = origin_data['special_tokens_mask']
        self.attention_mask = origin_data['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        sample = {
            'input_ids': self.input_ids[idx],
            'special_tokens_mask': self.special_tokens_mask[idx],
            'attention_mask': self.attention_mask[idx]
        }
        return sample

def get_data_collator(config):
    model_path = os.path.join(config.data_dir, 'model')
    tokenizer = LongformerTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    return data_collator

def build_train_sampler(config, dataset):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, seed = config.seed)
    else:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    return sampler

def build_train_dataloader(config):
    data_collator = get_data_collator(config)
    train_dataset = LongformerDataset(
        os.path.join(config.data_dir, 'dataset', 'train_dataset.npz'))
    train_sampler = build_train_sampler(config, train_dataset)
    data_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=data_collator,
            batch_size=config.train_batch_size,
            drop_last=config.dataloader_drop_last,
            num_workers=config.dataloader_num_workers,
        )
    return data_loader

def build_eval_dataloader(config):
    data_collator = get_data_collator(config)
    eval_dataset = LongformerDataset(
        os.path.join(config.data_dir, 'dataset', 'eval_dataset.npz'))
    data_loader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=config.eval_batch_size,
            drop_last=config.dataloader_drop_last,
            num_workers=config.dataloader_num_workers,
        )

    return data_loader

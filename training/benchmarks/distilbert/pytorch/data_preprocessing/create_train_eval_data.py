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
from datasets import load_dataset
from transformers import DistilBertTokenizer


def save_dataset(ds, save_path):
    np.savez(save_path,
             idx=ds['idx'],
             sentence=ds['sentence'],
             label=ds['label'],
             input_ids=ds['input_ids'],
             attention_mask=ds['attention_mask'],)


def main():
    data_prefix = 'distilbert/dataset'
    os.makedirs(data_prefix, exist_ok=True)
    train_datapath = os.path.join(data_prefix, 'train_dataset.npz')
    eval_datapath = os.path.join(data_prefix, 'eval_dataset.npz')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    raw_datasets = load_dataset("sst2")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"].with_format('numpy')
    save_dataset(train_dataset, train_datapath)

    eval_dataset = tokenized_datasets["validation"].with_format('numpy')
    save_dataset(eval_dataset, eval_datapath)


if __name__ == "__main__":
    main()

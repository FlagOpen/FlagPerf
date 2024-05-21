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
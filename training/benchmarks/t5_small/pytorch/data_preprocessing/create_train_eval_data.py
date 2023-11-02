import os

import numpy as np
import datasets
from transformers import AutoTokenizer


def save_dataset(ds, save_path):
    np.savez(save_path,
             input_ids=ds['input_ids'],
             attention_mask=ds['attention_mask'],
             labels=ds['labels'])


def main():
    data_prefix = 't5_small_train/dataset'
    os.makedirs(data_prefix, exist_ok=True)
    train_datapath = os.path.join(data_prefix, 'train_dataset.npz')
    eval_datapath = os.path.join(data_prefix, 'eval_dataset.npz')

    tokenizer = AutoTokenizer.from_pretrained('t5-small',
                                              use_fast=True,
                                              revision='main')

    raw_datasets = datasets.load_dataset('cnn_dailymail', '3.0.0')

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        text_column = 'article'
        summary_column = 'highlights'
        prefix = 'summarize: '
        max_source_length = 1024
        max_target_length = 128
        ignore_pad_token_for_loss = True
        padding = "max_length"

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs,
                                 max_length=max_source_length,
                                 padding=padding,
                                 truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets,
                           max_length=max_target_length,
                           padding=padding,
                           truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [[
                (l if l != tokenizer.pad_token_id else -100) for l in label
            ] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=32,
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    ).with_format('numpy')
    save_dataset(train_dataset, train_datapath)

    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=32,
        remove_columns=raw_datasets["validation"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    ).with_format('numpy')
    save_dataset(eval_dataset, eval_datapath)


if __name__ == "__main__":
    main()

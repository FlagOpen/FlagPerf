import os
import numpy as np
from itertools import chain
from typing import List
import datasets
from datasets import load_dataset
from transformers import LongformerTokenizer, DataCollatorForLanguageModeling

def save_dataset(ds, save_path):
    np.savez(save_path,
             input_ids=ds['input_ids'],
             attention_mask=ds['attention_mask'],
             special_tokens_mask=ds['special_tokens_mask'])

def main():
    data_prefix = 'longformer_train/dataset'
    os.makedirs(data_prefix, exist_ok=True)
    train_datapath = os.path.join(data_prefix, 'train_dataset.npz')
    eval_datapath = os.path.join(data_prefix, 'eval_dataset.npz')

    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', use_auto_token=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    max_seq_length = tokenizer.model_max_length
    if max_seq_length > 1024:
        max_seq_length = 1024

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=None,
                    remove_columns=[text_column_name],
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset line_by_line",
                )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=None,
                    load_from_cache_file=True,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
    train_dataset = tokenized_datasets['train'].with_format('numpy')
    save_dataset(train_dataset, train_datapath)

    validation_dataset = tokenized_datasets['validation'].with_format('numpy')
    save_dataset(validation_dataset, eval_datapath)

if __name__ == "__main__":
    main()

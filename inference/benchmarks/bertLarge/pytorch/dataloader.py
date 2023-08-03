from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import random


class BertInferDataset(Dataset):

    def __init__(self, input_ids, label_ids, random_dupo, seq_length):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.random_dupo = random_dupo
        self.seq_length = seq_length

    def __len__(self):
        return len(self.input_ids) // self.seq_length * self.random_dupo

    def __getitem__(self, idx):
        idx_global = idx // self.random_dupo
        start_idx = idx_global * self.seq_length
        chunk_input = self.input_ids[start_idx:start_idx + self.seq_length]
        chunk_label = self.label_ids[start_idx:start_idx + self.seq_length]

        chunk_input = torch.tensor(chunk_input).int()
        chunk_label = torch.tensor(chunk_label).int()

        return (chunk_input, chunk_label)


def build_dataset(config):

    random.seed(config.random_seed)

    with open(config.data_dir + "/" + config.eval_file, "r") as file:
        text = file.read()

    tokenizer = BertTokenizer.from_pretrained(config.data_dir + "/" +
                                              config.weight_dir)
    tokens = tokenizer.tokenize(text)

    label_ids = tokenizer.convert_tokens_to_ids(tokens)
    label_ids = [tokenizer.cls_token_id] + label_ids + [tokenizer.sep_token_id]

    masked_tokens = []
    for token in tokens:
        if token != "[CLS]" and token != "[SEP]":
            masked_tokens.append(
                "[MASK]" if random.random() < config.mask_ratio else token)
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    dataset = BertInferDataset(input_ids, label_ids, config.random_dupo,
                               config.seq_length)

    return dataset


def build_dataloader(config):
    dataset = build_dataset(config)
    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        drop_last=True,
                        num_workers=config.num_workers,
                        pin_memory=True)

    return loader

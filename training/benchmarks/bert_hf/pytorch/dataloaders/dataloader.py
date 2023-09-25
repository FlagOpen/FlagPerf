from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import random


class BertInferDataset(Dataset):

    def __init__(self, input_ids, label_ids, seq_length):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.seq_length = seq_length

    def __len__(self):
        return len(self.input_ids) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        chunk_input = self.input_ids[start_idx:start_idx + self.seq_length]
        chunk_label = self.label_ids[start_idx:start_idx + self.seq_length]

        chunk_input = torch.tensor(chunk_input).int()
        chunk_label = torch.tensor(chunk_label).long()

        return (chunk_input, chunk_label)


def build_dataset(config):

    random.seed(config.seed)

    with open(config.data_dir + "/" + config.dataset_file, "r") as file:
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

    train_len = int(len(input_ids) * (1.0 - config.train_val_ratio))

    trainset = BertInferDataset(input_ids[:train_len], label_ids[:train_len],
                                config.seq_length)
    valset = BertInferDataset(input_ids[train_len:], label_ids[train_len:],
                              config.seq_length)

    return trainset, valset


def build_dataloader(config):
    trainset, valset = build_dataset(config)

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

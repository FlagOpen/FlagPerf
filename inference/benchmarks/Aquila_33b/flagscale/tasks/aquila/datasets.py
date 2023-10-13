"""Aquila datasets."""

import json
import math

import numpy as np
import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer


def build_dataset(task):
    """Helper function to select and build dataset."""

    if task == 'AQUILA':
        return _build_aquila_dataset()

    raise NotImplementedError('dataset for {} task is not '
                              'implemented.'.format(task))


class _AquilaDataset(torch.utils.data.Dataset):

    def __init__(self, path, tokenizer, seq_len):
        print_rank_0('> building aquila dataset from {} ...'.format(path))
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.BOS_TOKEN = self.tokenizer.cls
        self.EOS_TOKEN = self.tokenizer.eod
        # 2048 for 7B
        self.text_maxlen = seq_len

        import jsonlines
        self.texts = []
        with jsonlines.open(path) as reader:
            for line in reader:
                if 'text' not in line:
                    continue
                text = line['text'][:self.text_maxlen]
                self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = [self.BOS_TOKEN]
        tokens += self.tokenizer.tokenize(text)
        tokens.append(self.EOS_TOKEN)
        tokens = tokens[:self.seq_len+1]
        num_tokens = len(tokens)
        pad_mask = [1] * num_tokens
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [0] * (num_pad)
        pad_mask = np.array(pad_mask[1:])
        tokens = np.array(tokens)

        return {'text': tokens, 'pad_mask': pad_mask}


def _build_aquila_dataset():
    """Build aquila dataset."""
    args = get_args()
    tokenizer = get_tokenizer()

    assert len(args.valid_data) == 1
    val_dataset = _AquilaDataset(args.valid_data[0], tokenizer,
                                 args.seq_length)
    print_rank_0(' > found {} samples.'.format(len(val_dataset)))

    return val_dataset

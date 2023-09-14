# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""

import json

import numpy as np
import torch

from dataloaders.indexed_dataset import make_dataset as make_indexed_dataset
from dataloaders.dataloader import build_data_loader
from dataloaders import get_tokenizer

import config

def build_train_test_datasets(train_num_samples,
                                seq_length, seed, skip_warmup,
                                train_data_prefix=None,
                                test_data_prefix=None):
    """Build train, valid, and test datasets."""
    # get the tokenizer
    tokenizer = get_tokenizer()

    train_dataset, test_dataset = None, None
    # Single dataset.
    assert train_data_prefix is not None
    train_dataset = build_dataset("train", train_data_prefix,
                                train_num_samples, seq_length, seed,
                                skip_warmup)
    assert test_data_prefix is not None
    test_dataset = _LambadaDataset(test_data_prefix, tokenizer.eod, tokenizer,
                                  seq_length)
    return (train_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, num_samples, seq_length, seed, skip_warmup):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """
    dataset = None

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]

    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    dataset = GPTDataset(dataset_name, data_prefix,
                        documents, indexed_dataset,
                        num_samples, seq_length, seed)

    return dataset


def get_indexed_dataset_(data_prefix, skip_warmup):
    """Build indexed dataset."""
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           "mmap",
                                           skip_warmup)
    return indexed_dataset


class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed):

        self.name = name
        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            documents, self.indexed_dataset.sizes, seq_length, seed)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return {'text': np.array(sample, dtype=np.int64)}


def _build_index_mappings(documents, sizes, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # doc-idx.
    doc_idx = _build_doc_idx(documents, np_rng)
    # sample-idx.
    assert doc_idx.dtype == np.int32
    assert sizes.dtype == np.int32
    sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
                                   tokens_per_epoch)
    # shuffle-idx.
    # -1 is due to data structure used to retieve the index:
    #    sample i --> [sample_idx[i], sample_idx[i+1])
    num_samples_ = sample_idx.shape[0] - 1
    shuffle_idx = _build_shuffle_idx(num_samples_,
                                        sample_idx.shape[0] - 1, np_rng)
    
    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _build_doc_idx(documents, np_rng):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    doc_idx = np.mgrid[0:1, 0:len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx



def _build_sample_idx(sizes, doc_idx, seq_length, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


def build_train_test_data_dataloaders(
        build_train_test_datasets_provider):
    """XXX"""

    (train_dataloader, test_dataloader) = (None, None)

    # Number of train/valid/test samples.
    train_samples = config.max_steps* config.global_batch_size

    # Build the datasets.
    train_ds, test_ds = build_train_test_datasets_provider(
        train_num_samples=train_samples)

    # Build dataloders.
    train_dataloader = build_data_loader(train_ds, config.train_batch_size,
                                   config.num_workers, drop_last=False)

    test_dataloader = build_data_loader(test_ds, config.train_batch_size,
                                   config.num_workers, drop_last=False)

    # Flags to know if we need to do training/validation/testing.
    config.do_train = train_dataloader is not None and config.max_steps> 0

    return train_dataloader, test_dataloader 


class _LambadaDataset(torch.utils.data.Dataset):

    def __init__(self, path, pad_idx, tokenizer, seq_len, strict=False):
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.strict = strict

        self.tokens = []
        self.labels = []
        with open(path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = self.get_tokens(text)
                self.tokens.append(tokens)
                self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.tokenize(text)
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.tokenize(text[:start_idx].strip())
        last_token = self.tokenizer.tokenize(' ' + last_token)
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        num_tokens = len(tokens)
        pad_mask = [0] * num_tokens
        labels = self.labels[idx]
        pad_mask += [1] * len(labels)
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])

        return {'text': np.array(tokens), 'pad_mask': pad_mask}


import os
import sys
import numpy as np
import torch
from dataloaders.samplers import DistributedBatchSampler, RandomSampler

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch


class GenDataset(torch.utils.data.Dataset):

    def __init__(self, args, data_path, split, tokenizer, ratio=1):
        self.split = split
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.args = args
        self.seq_length = args.seq_length

        self.pad_id = tokenizer.encoder['<pad>']
        self.eod_token = tokenizer.encoder['<eod>']
        args.eod_token = tokenizer.encoder['<eod>']

        with open(data_path, "r", encoding='utf-8') as f:
            data = f.readlines()
        self.samples = self.process(data)

    def process(self, data):
        samples = []
        for doc in data[:int(self.ratio * len(data))]:
            token_ids = self.tokenizer.encode(doc)
            token_ids.append(self.eod_token)
            start = 0

            while start + self.seq_length + 1 < len(token_ids):
                samples.append(token_ids[start:start + self.seq_length + 1])
                start = start + self.seq_length + 1
            samples.append(token_ids[start:] + [self.pad_id] *
                           (self.seq_length + 1 - (len(token_ids) - start)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate(self, samps):
        bs = len(samps)

        # triangle attention mask
        attn_mask = torch.tril(torch.ones(
            (self.seq_length, self.seq_length))).unsqueeze(0)
        position_ids = torch.arange(self.seq_length,
                                    dtype=torch.long).unsqueeze(0).repeat(
                                        bs, 1)

        if self.args.fp16:
            attn_mask = attn_mask.half()

        # the data that need to go through the model
        batch_sample = {
            "input_ids": torch.ones(bs, self.seq_length).long() * self.pad_id,
            "attention_mask": attn_mask.unsqueeze(1),
            "position_ids": position_ids,
        }

        # the data that do not need to go through the model
        no_model_sample = {
            "labels": torch.ones(bs, self.seq_length).long() * self.pad_id,
            "loss_mask": torch.zeros(bs, self.seq_length).float()
        }

        for i, samp in enumerate(samps):
            assert len(samp) == self.seq_length + 1, (len(samp),
                                                      self.seq_length)
            batch_sample["input_ids"][i] = torch.tensor(samp[:-1],
                                                        dtype=torch.long)
            no_model_sample["labels"][i] = torch.tensor(samp[1:],
                                                        dtype=torch.long)
            no_model_sample["loss_mask"][i] = (no_model_sample["labels"][i] !=
                                               self.pad_id).float()

        return batch_sample, no_model_sample


def check_md5_data_file(data_type, file_name):
    import hashlib

    def get_file_md5(fname):
        m = hashlib.md5()
        with open(fname, 'rb') as fobj:
            while True:
                data = fobj.read(4096)
                if not data:
                    break
                m.update(data)

        return m.hexdigest()

    official_md5_train = '65bdbb2b3a8d3b61dbe63ea1a67eec62'
    official_md5_valid = 'b7d8356e0d921b512f9b6860138f2174'

    file_md5 = get_file_md5(file_name)

    if data_type == 'train':
        return file_md5 == official_md5_train
    else:
        return file_md5 == official_md5_valid


def load_data(args, data_type, tokenizer, ratio=1):
    data_path = args.data_dir
    if data_type == 'train':
        batch_size = args.train_batch_size
    else:
        batch_size = args.eval_batch_size

    # Data parallel arguments.
    world_size = dist_pytorch.get_world_size()
    rank = dist_pytorch.get_rank()
    global_batch_size = batch_size * world_size
    num_workers = args.num_workers

    # Dataset
    filename = os.path.join(data_path, data_type + '.txt')
    dataset = GenDataset(args, filename, data_type, tokenizer, ratio=ratio)

    # Use a random sampler with distributed batch sampler.
    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       collate_fn=dataset.collate), dataset

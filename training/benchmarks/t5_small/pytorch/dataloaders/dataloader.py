import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class T5Dataset(Dataset):
    def __init__(self, filepath):
        origin_data = np.load(filepath)
        self.input_ids = origin_data['input_ids']
        self.attention_mask = origin_data['attention_mask']
        self.labels = origin_data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        sample = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        return sample


def _prepare_decoder_input_ids_from_labels(input_ids):
    """
        https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/t5/modeling_t5.py#L1800
        https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/t5/modeling_t5.py#L851
    """
    decoder_start_token_id = 0
    pad_token_id = 0

    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def my_collate(batch):
    """
        https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/data/data_collator.py#L600
    """
    new_batch = default_collate(batch)
    new_batch["decoder_input_ids"] = _prepare_decoder_input_ids_from_labels(
        new_batch["labels"])
    return new_batch


def build_train_dataloader(config):
    train_dataset = T5Dataset(
        os.path.join(config.data_dir, 'dataset', 'train_dataset.npz'))

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.train_batch_size,
        collate_fn=my_collate)
    return data_loader


def build_eval_dataloader(config):
    eval_dataset = T5Dataset(
        os.path.join(config.data_dir, 'dataset', 'eval_dataset.npz'))

    data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config.eval_batch_size, collate_fn=my_collate)
    return data_loader


if __name__ == '__main__':
    from collections import namedtuple
    Config = namedtuple(
        'Config',
        ['data_dir', 'distributed', 'train_batch_size', 'eval_batch_size'])
    config = Config('t5_small_train/dataset', False, 4, 4)
    eval_dataloader = build_eval_dataloader(config)
    for i, batch in enumerate(eval_dataloader):
        break

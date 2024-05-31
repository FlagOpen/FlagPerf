import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
InputDataClass = NewType("InputDataClass", Any)

class DistilBertDataset(Dataset):
    def __init__(self, filepath):
        origin_data = np.load(filepath)
        self.idx = origin_data['idx']
        self.sentence = origin_data['sentence']
        self.label = origin_data['label']
        self.input_ids = origin_data['input_ids']
        self.attention_mask = origin_data['attention_mask']

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        sample = {
            'sentence': self.sentence[idx],
            'label': self.label[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        return sample


def default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    """
        https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/data/data_collator.py#L105
    """
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def build_train_sampler(config, dataset):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, seed = config.seed)
    else:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    return sampler


def build_train_dataloader(config):
    train_dataset = DistilBertDataset(
        os.path.join(config.data_dir, 'dataset', 'train_dataset.npz'))

    train_sampler = build_train_sampler(config, train_dataset)
    data_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=config.train_batch_size,
            collate_fn=default_data_collator,
            drop_last=config.dataloader_drop_last,
            num_workers=config.dataloader_num_workers,
        )
    return data_loader


def build_eval_sampler(dataset):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    return sampler


def build_eval_dataloader(config):
    eval_dataset = DistilBertDataset(
        os.path.join(config.data_dir, 'dataset', 'eval_dataset.npz'))

    eval_sampler = build_eval_sampler(eval_dataset)
    data_loader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=config.eval_batch_size,
            collate_fn=default_data_collator,
            drop_last=config.dataloader_drop_last,
            num_workers=config.dataloader_num_workers,
        )
    
    return data_loader


if __name__ == '__main__':
    from collections import namedtuple
    Config = namedtuple(
        'Config',
        ['data_dir', 'distributed', 'train_batch_size', 'eval_batch_size', 'dataloader_drop_last', 'dataloader_num_workers', 'seed'])
    config = Config('distilbert', False, 4, 4, False, 8, 1234)
    train_dataloader = build_train_dataloader(config)
    for i, batch in enumerate(train_dataloader):
        print(batch.keys())
        break
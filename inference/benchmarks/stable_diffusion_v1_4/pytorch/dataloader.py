from torch.utils.data import DataLoader as dl
import torch
import json
import random


def build_dataset(config):

    df = json.load(open(config.data_dir + "/" + config.prompts))["annotations"]
    prompts = []
    for item in df:
        prompts.append(item["caption"])
    dataset = [
        item for item in prompts if len(item) < config.prompt_max_len - 2
    ]
    random.seed(config.random_seed)
    dataset = random.sample(dataset, config.prompt_samples)

    return dataset


def build_dataloader(config):
    dataset = build_dataset(config)
    loader = dl(dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=config.num_workers,
                pin_memory=True)

    return loader

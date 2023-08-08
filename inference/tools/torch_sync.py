import torch


def torch_sync(config):
    if config.vendor == "nvidia":
        torch.cuda.synchronize()

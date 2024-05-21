from driver import dist_pytorch
import torch
import config


def convert_model(model):
    model.to(memory_format=torch.channels_last)
    return model

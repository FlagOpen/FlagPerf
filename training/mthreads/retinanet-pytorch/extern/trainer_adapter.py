from driver import dist_pytorch
import torch
import config


def convert_model(model):
    if config.nhwc:
        if dist_pytorch.get_rank() == 0:
            print("convert nhwc model", flush=True)
        model.to(memory_format=torch.channels_last)
    return model

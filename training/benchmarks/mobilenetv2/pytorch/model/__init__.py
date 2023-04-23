import torch
import torchvision


def create_model(config):
    model = torchvision.models.mobilenet_v2()
    return model

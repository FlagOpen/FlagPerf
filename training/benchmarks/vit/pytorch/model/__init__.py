import torch
import torchvision


def create_model(config):
    model = torchvision.models.vit_b_16()
    return model

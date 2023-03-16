import torch
import torchvision

def create_model(config):
    #model = torchvision.models.mobilenet_v2()
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
    return model

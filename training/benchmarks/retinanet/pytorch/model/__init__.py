from torchvision.models.detection import retinanet_resnet50_fpn
import torchvision

import os

def create_model():
    torchvision.models.resnet.__dict__['model_urls'][
        'resnet50'] = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    return retinanet_resnet50_fpn()

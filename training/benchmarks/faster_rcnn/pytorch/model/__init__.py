from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
from packaging import version


def create_model():

    rn50_backbone_url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    if version.parse(torchvision.__version__) > version.parse("0.12.0"):
        torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1.value.url = rn50_backbone_url
    else:
        torchvision.models.resnet.__dict__['model_urls']['resnet50'] = rn50_backbone_url
    return fasterrcnn_resnet50_fpn()

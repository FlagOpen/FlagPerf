from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
from packaging.version import Version

def create_model():
    TORCHVISION_VERSION = Version(torchvision.__version__).base_version
    if Version(TORCHVISION_VERSION) > Version("0.12.0"):
        # model_urls has gone since v0.13.0
        torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1.value.url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    else:
        torchvision.models.resnet.__dict__['model_urls'][
            'resnet50'] = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    return fasterrcnn_resnet50_fpn()

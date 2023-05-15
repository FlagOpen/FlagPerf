from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision


def create_model():

    torchvision.models.resnet.__dict__['model_urls'][
        'resnet50'] = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    return fasterrcnn_resnet50_fpn()

from torchvision.models.detection import fasterrcnn_resnet50_fpn


def create_model():
    return fasterrcnn_resnet50_fpn()

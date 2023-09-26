from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights as w


def create_model(config):
    if config.no_validation:
        assert config.exist_onnx_path is not None
        return None
    model = resnet50(weights=w.IMAGENET1K_V1)
    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model

from torchvision.models import swin_b
from torchvision.models import Swin_B_Weights as w


def create_model(config):
    if config.no_validation:
        assert config.exist_onnx_path is not None
        return None
    model = swin_b(weights=w.IMAGENET1K_V1)
    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model

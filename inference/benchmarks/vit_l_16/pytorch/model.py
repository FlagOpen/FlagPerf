from transformers import ViTForImageClassification as vit


def create_model(config):
    if config.no_validation:
        assert config.exist_onnx_path is not None
        return None
    model = vit.from_pretrained(config.weights)
    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model

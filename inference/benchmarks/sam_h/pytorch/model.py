from .model_utils.sam import SamModel


def create_model(config):
    if config.no_validation:
        assert config.exist_onnx_path is not None
        return None
    model = SamModel.from_pretrained(config.data_dir + "/" + config.weights)
    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model

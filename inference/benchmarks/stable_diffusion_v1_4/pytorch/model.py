from .model_utils.unet2d import UNet2DConditionModel


def create_model(config):
    if config.no_validation:
        assert config.exist_onnx_path is not None
        return None
    model = UNet2DConditionModel.from_pretrained(config.data_dir + "/" +
                                                 config.weights,
                                                 subfolder="unet")
    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model

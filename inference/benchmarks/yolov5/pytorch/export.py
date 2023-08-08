def export_model(model, config):
    if config.exist_onnx_path is not None:
        return config.exist_onnx_path
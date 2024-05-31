def create_model(config):
    if config.no_validation:
        assert config.exist_onnx_path is not None
        return None

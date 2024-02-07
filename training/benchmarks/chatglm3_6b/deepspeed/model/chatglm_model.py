from transformers import AutoModel, AutoConfig


def get_chatglm_model(model_config_dir, flashattn):

    config = AutoConfig.from_pretrained(model_config_dir, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True, empty_init=False)

    return model

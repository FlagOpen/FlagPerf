from transformers import AutoModelForCausalLM


def get_baichuan_model(model_config_dir):

    model = AutoModelForCausalLM.from_pretrained(model_config_dir, trust_remote_code=True)

    return model


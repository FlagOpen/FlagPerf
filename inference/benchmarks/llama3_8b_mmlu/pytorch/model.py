from transformers import AutoModelForCausalLM
import os


def create_model(config):
    model_path = os.path.join(config.data_dir, config.weight_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_path).eval().cuda().float()

    if config.fp16:
        model.half()

    return model

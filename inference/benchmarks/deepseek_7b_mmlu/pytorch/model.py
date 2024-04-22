from transformers import AutoModelForCausalLM


def create_model(config):
    model = AutoModelForCausalLM.from_pretrained(config.data_dir + "/" +
                                            config.weight_dir).eval().cuda().float()

    if config.fp16:
        model.half()

    return model

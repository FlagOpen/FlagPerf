from transformers import LlamaForCausalLM


def create_model(config):
    model = LlamaForCausalLM.from_pretrained(config.data_dir + "/" +
                                            config.weight_dir).eval().cuda().float()

    if config.fp16:
        model.half()

    return model

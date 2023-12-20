from transformers import LlamaForCausalLM, LlamaConfig


def get_llama_model(model_config_dir, flashattn):

    config = LlamaConfig.from_pretrained(model_config_dir)
    config._flash_attn_2_enabled = flashattn
    model = LlamaForCausalLM(config)

    return model

from transformers import (AutoConfig,
                          AutoModelForMaskedLM,
                          AutoTokenizer)


def create_model(config):
    model_name_or_path = "allenai/longformer-base-4096"
    hfconfig = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            low_cpu_mem_usage=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, hfconfig, tokenizer

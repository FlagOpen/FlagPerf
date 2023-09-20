import os
from transformers import (AutoConfig,
                          AutoModelForMaskedLM,
                          AutoTokenizer)


def create_model(config):
    model_path = os.path.join(config.data_dir, 'model')
    hfconfig = AutoConfig.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            from_tf=False,
            config=hfconfig,
            low_cpu_mem_usage=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, hfconfig, tokenizer

import os
from transformers import (AutoConfig,
                          LongformerForMaskedLM,
                          LongformerTokenizerFast,
                          AutoModelForMaskedLM,
                          AutoTokenizer)


def create_model(config):
    model_path = os.path.join(config.data_dir, 'model')
    hfconfig = AutoConfig.from_pretrained(model_path)
    model = LongformerForMaskedLM(hfconfig)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return model, hfconfig, tokenizer

import os
from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast


def create_model(config):
    model_path = os.path.join(config.data_dir, 'model')
    hfconfig = T5Config.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path,
                                                       config=hfconfig)
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    return model, hfconfig, tokenizer


if __name__ == '__main__':

    from collections import namedtuple
    Config = namedtuple('Config', ['data_dir'])
    config = Config('t5_small_train')
    model, tokenizer = create_model(config)

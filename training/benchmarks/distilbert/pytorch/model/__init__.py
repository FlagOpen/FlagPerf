import os
from collections import namedtuple

from transformers import DistilBertConfig, DistilBertTokenizer
from transformers import DistilBertForSequenceClassification


def create_model(config):
    model_path = os.path.join(config.data_dir, 'model')
    hfconfig = DistilBertConfig.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path,
                                                       config=hfconfig)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    return model, hfconfig, tokenizer


if __name__ == '__main__':

    Config = namedtuple('Config', ['data_dir'])
    config = Config('distilbert')
    model, model_config, tokenizer = create_model(config)
    import pdb; pdb.set_trace()
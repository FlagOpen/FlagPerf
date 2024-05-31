from transformers import BertForMaskedLM


def create_model(config):
    model = BertForMaskedLM.from_pretrained(config.data_dir + "/" +
                                            config.weight_dir)
    return model

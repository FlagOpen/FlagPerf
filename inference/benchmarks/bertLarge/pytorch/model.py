from transformers import BertForMaskedLM


def create_model(config):
    model = BertForMaskedLM.from_pretrained(config.data_dir + "/" +
                                            config.weight_dir,
                                            torchscript=True)
    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model

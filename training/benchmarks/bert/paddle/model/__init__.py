from model.models.modeling import BertConfig, BertForPreTraining


def create_model(config):
    config.resume_step = 0

    bert_config = BertConfig.from_json_file(config.bert_config_path)
    bert_config.fused_gelu_bias = config.fused_gelu_bias
    bert_config.dense_seq_output = config.dense_seq_output
    bert_config.fuse_dropout = config.enable_fuse_dropout
    bert_config.fused_dropout_add = config.fused_dropout_add

    # Padding for divisibility by 8
    if bert_config.vocab_size % 8 != 0:
        bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)

    model = BertForPreTraining(bert_config)
    return bert_config, model
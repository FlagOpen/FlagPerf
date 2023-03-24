from collections import OrderedDict

from driver import dist_pytorch
from .layers.transformer import GPT2Transformer


def convert_model(model, config):
    if dist_pytorch.get_rank() == 0:
        print("use apex layer norm", flush=True)
    state_dict = model.state_dict()
    transformer_layer = GPT2Transformer(num_layers=config.num_layers,
                                                hidden_size=config.hidden_size,
                                                num_attention_heads=config.num_attention_heads,
                                                max_sequence_length=config.max_seq_length,
                                                max_memory_length=config.max_memory_length,
                                                embedding_dropout_prob=config.hidden_dropout,
                                                attention_dropout_prob=config.attention_dropout,
                                                output_dropout_prob=config.hidden_dropout,
                                                checkpoint_activations=config.checkpoint_activations)
    model.model.transformer = transformer_layer
    state_dict = remap_attn_parameters(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model


def remap_attn_parameters(model_dict):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'attention' in k:
            if 'attention.query_key_value.weight' in k:
                data = model_dict[k]
                q_w, k_w, v_w = data.chunk(3, dim=0)
                new_q_k = k.replace(
                    'attention.query_key_value.weight', 'attention.layer.q_weight')
                new_k_k = k.replace(
                    'attention.query_key_value.weight', 'attention.layer.k_weight')
                new_v_k = k.replace(
                    'attention.query_key_value.weight', 'attention.layer.v_weight')
                res_dict[new_q_k] = q_w
                res_dict[new_k_k] = k_w
                res_dict[new_v_k] = v_w
            elif 'attention.query_key_value.bias' in k:
                data = model_dict[k]
                q_w, k_w, v_w = data.chunk(3, dim=0)
                new_q_k = k.replace(
                    'attention.query_key_value.bias', 'attention.layer.q_bias')
                new_k_k = k.replace(
                    'attention.query_key_value.bias', 'attention.layer.k_bias')
                new_v_k = k.replace(
                    'attention.query_key_value.bias', 'attention.layer.v_bias')
                res_dict[new_q_k] = q_w
                res_dict[new_k_k] = k_w
                res_dict[new_v_k] = v_w
            elif 'attention.dense.weight' in k:
                new_k = k.replace('attention.dense.weight',
                                  'attention.layer.out_proj_weight')
                res_dict[new_k] = model_dict[k]
            elif 'attention.dense.bias' in k:
                new_k = k.replace('attention.dense.bias',
                                  'attention.layer.out_proj_bias')
                res_dict[new_k] = model_dict[k]
            else:
                res_dict[k] = model_dict[k]
        # elif "dense_h_to_4h.weight" in k:
        #     new_k = k.replace('dense_h_to_4h.weight',
        #                           'fused_dense.weight1')
        #     res_dict[new_k] = model_dict[k]
        # elif "dense_h_to_4h.bias" in k:
        #     new_k = k.replace('dense_h_to_4h.bias',
        #                           'fused_dense.bias1')
        #     res_dict[new_k] = model_dict[k]
        # elif "dense_4h_to_h.weight" in k:
        #     new_k = k.replace('dense_4h_to_h.weight',
        #                           'fused_dense.weight2')
        #     res_dict[new_k] = model_dict[k]
        # elif "dense_4h_to_h.bias" in k:
        #     new_k = k.replace('dense_4h_to_h.bias',
        #                           'fused_dense.bias2')
        #     res_dict[new_k] = model_dict[k]
        else:
            res_dict[k] = model_dict[k]
    model_dict.clear()
    return res_dict

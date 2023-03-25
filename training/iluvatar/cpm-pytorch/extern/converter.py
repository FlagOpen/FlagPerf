from collections import OrderedDict
from .layers import LayerNorm


def convert_model(model, config):
    state_dict = model.state_dict()
    # state_dict = remap_attn_parameters(state_dict)

    for i in range(config.num_layers):
        model.transformer.layers[i].input_layernorm = LayerNorm(config.hidden_size, config.layernorm_epsilon)
        model.transformer.layers[i].post_attention_layernorm = LayerNorm(config.hidden_size, config.layernorm_epsilon)

    model.transformer.final_layernorm = LayerNorm(config.hidden_size, config.layernorm_epsilon)

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
        else:
            res_dict[k] = model_dict[k]
    model_dict.clear()
    return res_dict

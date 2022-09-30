from .layers import LayerNorm

def convert_model(model, config):
    state_dict = model.state_dict()
    state_dict = remap_attn_parameters(state_dict)

    for i in range(config.num_layers):
        model.transformer.layers[i].input_layernorm = LayerNorm(config.hidden_size, config.layernorm_epsilon)
        model.transformer.layers[i].post_attention_layernorm = LayerNorm(config.hidden_size, config.layernorm_epsilon)
    model.transformer.final_layernorm = LayerNorm(config.hidden_size, config.layernorm_epsilon)
    
    model.load_state_dict(state_dict, strict=True)
    return model


def remap_attn_parameters(model_dict):
    return model_dict

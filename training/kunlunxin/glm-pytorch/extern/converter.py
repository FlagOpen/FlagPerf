import torch

from driver import dist_pytorch
from .layers.transformer import GLMTransformer


def convert_model(model, config):
    if dist_pytorch.get_rank() == 0:
        print("convert model", flush=True)
    model = model.cpu()
    state_dict = model.state_dict()
    # the initilization of GLMTransformer will change the rng state
    with torch.random.fork_rng():
        transformer_layer = GLMTransformer(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=config.max_seq_length,
            max_memory_length=config.max_memory_length,
            embedding_dropout_prob=config.hidden_dropout,
            attention_dropout_prob=config.attention_dropout,
            output_dropout_prob=config.hidden_dropout,
            checkpoint_activations=config.checkpoint_activations)
    model.model.transformer = transformer_layer
    model.load_state_dict(state_dict, strict=True)
    model = model.to(config.device)
    return model

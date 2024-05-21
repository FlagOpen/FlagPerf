import torch

from .models.modeling import GLMModel, GLMForMultiTokenCloze, FP16_Module


def create_model(config):
    model = GLMModel(num_layers=config.num_layers,
                     vocab_size=config.vocab_size,
                     hidden_size=config.hidden_size,
                     num_attention_heads=config.num_attention_heads,
                     embedding_dropout_prob=config.hidden_dropout,
                     attention_dropout_prob=config.attention_dropout,
                     output_dropout_prob=config.hidden_dropout,
                     max_sequence_length=config.max_position_embeddings,
                     max_memory_length=config.max_memory_length,
                     checkpoint_activations=config.checkpoint_activations)

    model = GLMForMultiTokenCloze(model)
    return model

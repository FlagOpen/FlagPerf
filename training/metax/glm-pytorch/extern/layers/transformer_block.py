import torch

from model.layers.attention import SelfAttention
from .layernorm import LayerNorm
from model.layers.mlp import GLMMLP


class GLMTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 relative_encoding=False,
                 performer=False,
                 attention_scale=1.0):
        super(GLMTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            relative_encoding=relative_encoding,
            performer=performer,
            attention_scale=attention_scale)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = GLMMLP(hidden_size,
                          output_dropout_prob,
                          init_method,
                          output_layer_init_method=output_layer_init_method)

    def forward(self,
                hidden_states,
                ltor_mask,
                position_embeddings=None,
                r_w_bias=None,
                r_r_bias=None,
                mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask,
                                          position_embeddings, r_w_bias,
                                          r_r_bias, mem)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


if __name__ == "__main__":
    batch_size = 8
    seq_len = 512
    hidden_size = 1024
    num_attention_heads = 16
    attention_dropout_prob = 0.1
    output_dropout_prob = 0.1
    layernorm_epsilon = 1e-10
    init_method = torch.nn.init.xavier_normal_
    test_transformer = GLMTransformerLayer(hidden_size, num_attention_heads,
                                           attention_dropout_prob,
                                           output_dropout_prob,
                                           layernorm_epsilon, init_method)

    hidden_states = torch.rand([batch_size, seq_len, hidden_size])
    ltor_mask = torch.ones([1, 1, seq_len, seq_len])

    outputs = test_transformer(hidden_states, ltor_mask)
    print(outputs.shape)

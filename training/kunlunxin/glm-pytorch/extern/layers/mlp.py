import torch

import torch.nn as nn

from model.layers.linear import ColumnLinear, RowLinear
from torch.nn.functional import gelu


class GLMMLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 output_dropout_prob,
                 init_method,
                 output_layer_init_method=None):
        super(GLMMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnLinear(hidden_size,
                                          4 * hidden_size,
                                          gather_output=False,
                                          init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowLinear(4 * hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    hidden_size = 1024
    seq_len = 512
    batch_size = 8
    test_mlp = GLMMLP(hidden_size, 0.1, torch.nn.init.xavier_normal_)
    inputs = torch.rand([batch_size, seq_len, hidden_size])
    outputs = test_mlp(inputs)
    print(outputs.shape)

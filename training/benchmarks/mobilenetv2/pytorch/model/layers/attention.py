import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import contextlib
import math

from model.layers.linear import RowLinear, ColumnLinear
from torch.nn import Linear


def split_tensor_along_last_dim(tensor,
                                num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT2.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 init_method,
                 output_layer_init_method=None,
                 relative_encoding=False,
                 performer=False,
                 attention_scale=1.0):
        super(SelfAttention, self).__init__()
        self.performer = performer
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.

        self.hidden_size_per_partition = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)

        self.num_attention_heads_per_partition = num_attention_heads
        self.attention_scale = attention_scale
        # Strided linear layer.
        self.query_key_value = ColumnLinear(hidden_size,
                                            3 * hidden_size,
                                            stride=3,
                                            gather_output=False,
                                            init_method=init_method)

        if relative_encoding:
            self.relative = ColumnLinear(hidden_size,
                                         hidden_size,
                                         gather_output=False,
                                         init_method=init_method)

        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               input_is_parallel=True,
                               init_method=output_layer_init_method)

        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        # ql x kl x bsz x h
        # bsz x h x ql x kl
        zero_pad = torch.zeros((*x.size()[:-2], x.size(-2), 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self,
                hidden_states,
                ltor_mask,
                position_embeddings=None,
                r_w_bias=None,
                r_r_bias=None,
                mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        query_length = hidden_states.size(1)

        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer, mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(
                 mixed_x_layer, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer, mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(
                 mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        if self.attention_scale > 1.0:
            # Raw attention scores. [b, np, s, s]
            attention_scores = torch.matmul(
                query_layer / math.sqrt(self.attention_scale),
                key_layer.transpose(-1, -2) /
                math.sqrt(self.hidden_size_per_attention_head *
                          self.attention_scale))
        else:
            attention_scores = torch.matmul(
                query_layer,
                key_layer.transpose(-1, -2) /
                math.sqrt(self.hidden_size_per_attention_head))

        # Apply the left to right attention mask.
        attention_scores = torch.mul(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(dim=-1,
                                                        keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale
        # if torch.distributed.get_rank() == 0:
        #     print(min_attention_scores, attention_scores.max().item())
        attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # with get_cuda_rng_tracker().fork():
        #    attention_probs = self.attention_dropout(attention_probs)
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


if __name__ == "__main__":
    max_seq_length: int = 512
    num_layers = 24
    hidden_size = 1024
    num_attention_heads = 16
    hidden_dropout = 0.1
    attention_dropout = 0.1
    max_position_embeddings = 512
    mem_length = 0
    checkpoint_num_layers = 1
    attention_scale = 1
    vocab_size = 30592
    test_att = SelfAttention(hidden_size, num_attention_heads,
                             attention_dropout, attention_dropout,
                             init.xavier_normal_, init.xavier_normal_, False,
                             True, 1.0)
    input_2 = torch.rand([2, max_seq_length, hidden_size])
    mask = torch.ones([1, 1, max_seq_length, max_seq_length])
    output = test_att(input_2, mask)
    print(output.shape)

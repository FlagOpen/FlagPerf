
import torch
from torch import nn
from torch.nn import Parameter

from self_multihead_attn_func import self_attn_func
from fast_self_multihead_attn_func import fast_self_attn_func


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, impl='fast'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = bias
        self.impl = impl
        self.scaling = self.head_dim**-0.5

        self.q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.k_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.v_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.q_bias = Parameter(torch.Tensor(embed_dim))
        self.k_bias = Parameter(torch.Tensor(embed_dim))
        self.v_bias = Parameter(torch.Tensor(embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.reset_parameters()

        if impl == 'fast':
            self.attn_func = fast_self_attn_func
        elif impl == 'default':
            self.attn_func = self_attn_func
        else:
            assert False, "Unsupported impl: {} !".format(impl)

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.q_weight)
        nn.init.xavier_uniform_(self.k_weight)
        nn.init.xavier_uniform_(self.v_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)

        nn.init.constant_(self.q_bias, 0.)
        nn.init.constant_(self.k_bias, 0.)
        nn.init.constant_(self.v_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)

    def forward(self, query, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        mask = attn_mask
        input_weights = torch.cat([self.q_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim), self.k_weight.view(self.num_heads, 1, self.head_dim,
                                                                                                                            self.embed_dim), self.v_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim)], dim=1).reshape(3*self.embed_dim, self.embed_dim).contiguous()

        input_bias = torch.cat([self.q_bias.view(self.num_heads, 1, self.head_dim), self.k_bias.view(
            self.num_heads, 1, self.head_dim), self.v_bias.view(self.num_heads, 1, self.head_dim)], dim=1).reshape(3*self.embed_dim).contiguous()

        if self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query,
                                     input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask, False, self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, query,
                                     input_weights, self.out_proj_weight,
                                     input_bias, self.out_proj_bias,
                                     mask, False, self.dropout)

        return outputs

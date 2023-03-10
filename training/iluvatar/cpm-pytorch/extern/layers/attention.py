from torch.nn import MultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from layers.self_multihead_attn import SelfMultiheadAttn
# from apex.contrib.multihead_attn import SelfMultiheadAttn
# this is pytorch official， mask is same with paper

import time


class OfficialSelfAttention(torch.nn.Module):

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob):
        super(OfficialSelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.layer = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout_prob,
            bias=True,
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, ltor_mask, *args, **kwargs):

        hidden_states = hidden_states.transpose(0, 1)

        mask = (-65504.0) * (1.0 - ltor_mask)
        # mask = (1.0 - ltor_mask).bool()
        mask = mask.repeat([1, self.num_attention_heads, 1, 1])
        mask = mask.view([-1, mask.shape[-2], mask.shape[-1]])
        output = self.layer(hidden_states,
                            hidden_states,
                            hidden_states,
                            attn_mask=mask,
                            need_weights=False)
        output = output[0]
        output = output.transpose(0, 1)
        output = self.output_dropout(output)
        return output


# apex  official
class OfficialSelfAttentionApex(torch.nn.Module):

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob):
        super(OfficialSelfAttentionApex, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.layer = SelfMultiheadAttn(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout_prob,
            bias=True,
            impl="fast",  # fast or default
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, ltor_mask, *args, **kwargs):

        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # mask = (-65504.0) * (1.0 - ltor_mask)
        mask = 1 - ltor_mask
        mask = mask.byte()
        mask = mask.repeat([1, self.num_attention_heads, 1, 1])
        mask = mask.view([-1, mask.shape[-2], mask.shape[-1]])
        output = self.layer(hidden_states,
                            attn_mask=mask,
                            is_training=self.training)
        output = output.transpose(0, 1)
        output = self.output_dropout(output)
        return output

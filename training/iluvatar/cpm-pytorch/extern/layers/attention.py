# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.nn import MultiheadAttention
import torch
from layers.self_multihead_attn import SelfMultiheadAttn
# from apex.contrib.multihead_attn import SelfMultiheadAttn
# this is pytorch official， mask is same with paper


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

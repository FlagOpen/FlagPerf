# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.fairseq.modules.multihead_attention import RotaryEmbedding


def mha_state_dict_to_fairseq(sd):
    """Concatenate q, k, v matrices and load as usual."""
    new_sd = {}
    qkv = defaultdict(dict)

    for key, val in sd.items():
        fields = key.split('.')
        if len(fields) < 2:
            continue
        prefix = '.'.join(fields[:-2] + [""])
        module, param = fields[-2:]

        if module in ['q_proj', 'k_proj', 'v_proj']:
            qkv[prefix][module + '.' + param] = val
        else:
            new_sd[key] = val

    for prefix, param_dict in qkv.items():
        # Stitch qkv params together
        assert len(param_dict) == 6
        new_sd[f"{prefix}qkv.weight"] = torch.cat(
            [param_dict[f"{k}_proj.weight"] for k in ["q", "k", "v"]], dim=0)
        new_sd[f"{prefix}qkv.bias"] = torch.cat(
            [param_dict[f"{k}_proj.bias"] for k in ["q", "k", "v"]], dim=0)

    return new_sd


class PytMultiheadAttention(nn.Module):
    """Drop-in replacement for Fairseq MHA.

    Calls torch.nn.functional with combined qkv.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        self_attention=True,
        rotary_embeddings=False,
    ):
        super().__init__()

        assert self_attention
        assert not rotary_embeddings, "Not yet supported"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rotary_embeddings = rotary_embeddings

        if self.rotary_embeddings:
            self.rotary_freq = RotaryEmbedding(embed_dim)

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim
                ), "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim,
                             3 * num_heads * self.head_dim,
                             bias=bias)
        self.dropatt = nn.Dropout(dropout)
        self.out_proj = nn.Linear(num_heads * self.head_dim,
                                  embed_dim,
                                  bias=bias)
        self.reset_parameters()

        def hook(state_dict, prefix, *args, **kwargs):
            this_keys = {k for k in state_dict.keys() if k.startswith(prefix)}
            new_sd = {k: v for k, v in state_dict.items() if k in this_keys}
            for k in this_keys:
                del state_dict[k]
            state_dict.update(mha_state_dict_to_fairseq(new_sd))

        self._register_load_state_dict_pre_hook(hook)

    def forward(self,
                query,
                key=None,
                value=None,
                key_padding_mask=None,
                attn_mask=None):

        return F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.qkv.weight,
            self.qkv.bias,
            None,
            None,
            False,
            self.dropatt.p,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            average_attn_weights=False,
        )

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """Split q, k, v matrices for bwd compatibility with Fairseq."""
        sd = super().state_dict(*args, destination, prefix, keep_vars)
        for key in list(sd.keys()):
            if not (key.endswith(".qkv.weight") or key.endswith(".qkv.bias")):
                continue
            *pref, qkv, param = key.split(".")
            pref = ".".join(pref)
            assert qkv == "qkv"
            q, k, v = torch.chunk(sd.pop(key), 3, dim=0)
            sd[f"{pref}.q_proj.{param}"] = q
            sd[f"{pref}.k_proj.{param}"] = k
            sd[f"{pref}.v_proj.{param}"] = v

        return sd

    def reset_parameters(self):
        # Init as in Fairseq with qkv_same_dim=True and separate qkv projs
        t = self.qkv.weight.size(0) // 3
        nn.init.xavier_uniform_(self.qkv.weight[0 * t:1 * t],
                                gain=1 / (2**0.5))
        nn.init.xavier_uniform_(self.qkv.weight[1 * t:2 * t],
                                gain=1 / (2**0.5))
        nn.init.xavier_uniform_(self.qkv.weight[2 * t:3 * t],
                                gain=1 / (2**0.5))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

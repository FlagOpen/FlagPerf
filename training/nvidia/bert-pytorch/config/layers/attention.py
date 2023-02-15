from torch import nn

import apex
from apex.contrib.multihead_attn import SelfMultiheadAttn
from model.models.modeling import jit_dropout_add, BertSelfOutput
from .fmha import FMHA
from .mha import FastUnpadBertSelfAttention


#apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
import apex.normalization
#apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
from apex.contrib.layer_norm import FastLayerNorm as BertLayerNorm


# This module uses Apex C++ multihead attention implementation with fusions.
class FastBertAttention(nn.Module):
    def __init__(self, config):
        super(FastBertAttention, self).__init__()
        self.multi_head_attention = SelfMultiheadAttn(config.hidden_size, config.num_attention_heads, dropout = config.attention_probs_dropout_prob, bias=True, include_norm_add=False, impl='fast', separate_qkv_params=True, mask_additive=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.p = config.hidden_dropout_prob
        self.fused_dropout_add = config.fused_dropout_add
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, input_tensor, attention_mask, *args, **kwargs):
        residual=input_tensor
        multi_head_attention_output,_ = self.multi_head_attention(query = input_tensor, key = input_tensor, value = input_tensor, key_padding_mask=attention_mask, need_weights=True,attn_mask = None, is_training = self.training)
        if self.fused_dropout_add:
            attention_output = jit_dropout_add(multi_head_attention_output, residual, self.p, self.training)
            attention_output = self.layer_norm(attention_output)
            return attention_output
        else:
            attention_output = self.dropout(multi_head_attention_output)
            attention_output = self.layer_norm(attention_output + residual)
            return attention_output


class FastUnpadBertAttention(nn.Module):
    def __init__(self, config):
        super(FastUnpadBertAttention, self).__init__()
        if config.unpad_fmha:
            self.self = FMHA(config)
        else:
            self.self = FastUnpadBertSelfAttention(config, enable_stream=config.enable_stream, enable_sync=False, fuse_mask=config.fuse_mask, fuse_scale=config.fuse_scale, fuse_qkv=config.fuse_qkv, fuse_dropout=config.fuse_dropout, apex_softmax=config.apex_softmax, pad=config.pad)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, seqlen, batch):
        self_output = self.self(input_tensor, attention_mask, seqlen, batch, is_training = self.training)
        attention_output = self.output(self_output, input_tensor)
        return attention_output
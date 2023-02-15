import copy
import math
from torch.utils import checkpoint
import config

from model.models.modeling import BertAttention, BertIntermediate, BertOutput, BertForPreTraining

try:
    from .layers import *
except:
    from layers import *


class NvidiaBertLayer(nn.Module):
    def __init__(self, config):
        super(NvidiaBertLayer, self).__init__()
        self.unpad = config.unpad
        if config.fused_mha:
            self.attention = FastBertAttention(config)
        elif config.unpad:
            self.attention = FastUnpadBertAttention(config)
        else:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, seqlen=None, batch=None):
        attention_output = self.attention(hidden_states, attention_mask, seqlen, batch)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class NvidiaBertEncoder(nn.Module):
    def __init__(self, config):
        super(NvidiaBertEncoder, self).__init__()
        layer = NvidiaBertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.num_attention_heads = config.num_attention_heads
        self.fused_mha=config.fused_mha
        self.unpad=config.unpad
        self.unpad_fmha = config.unpad_fmha
        self.pad = config.pad
        self.fuse_mask = config.fuse_mask
        self.enable_stream = config.enable_stream

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, checkpoint_activations=False):
        # Unpad inputs and mask. It will remove tokens that are padded. Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens. Then unpadding performs the following compression of the inputs:
        #        hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
        batch = None
        seqlen = None
        if self.unpad:
            batch = hidden_states.shape[0]
            maxseqlen = hidden_states.shape[1]
            hidden_size = hidden_states.shape[2]
            attention_indices, attention_mask, seqlen, ntokens, cu_seqlens, actual_seqlens, maxseqlen_in_batch = generate_mask(attention_mask, self.num_attention_heads, pad=self.pad, fuse_mask=self.fuse_mask, unpad_fmha=self.unpad_fmha)
            if self.pad == True and self.enable_stream == False:
                hidden_states = hidden_states.view(batch,maxseqlen,hidden_size).permute(1,0,2).contiguous().view(batch*maxseqlen,hidden_size).contiguous()
            if self.pad == True and self.enable_stream == True:
                hidden_states = hidden_states.view(batch*maxseqlen,hidden_size)
            if self.pad == False:
                hidden_states = UnpadInput.apply(hidden_states.view(batch*maxseqlen, hidden_size).contiguous(), attention_indices, batch, maxseqlen, hidden_size, ntokens)

        all_encoder_layers = []
        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(custom(l, l+chunk_length), hidden_states, attention_mask*1)
                l += chunk_length
            # decoder layers
        else:
            if self.fused_mha:
                hidden_states = hidden_states.permute(1,0,2).contiguous()
            for i,layer_module in enumerate(self.layer):
                if seqlen is None and batch is None:
                    hidden_states = layer_module(hidden_states, attention_mask)
                else:
                    assert seqlen is not None
                    assert batch is not None
                    if self.unpad_fmha:
                        hidden_states = layer_module(hidden_states, cu_seqlens, actual_seqlens, maxseqlen_in_batch)
                    else:
                        hidden_states = layer_module(hidden_states, attention_mask, seqlen, batch)

                if output_all_encoded_layers:
                    if self.fused_mha:
                        all_encoder_layers.append(hidden_states.permute(1,0,2).contiguous())
                    else:
                        all_encoder_layers.append(hidden_states)

        # Pad inputs and mask. It will insert back zero-padded tokens. Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens. Then padding performs the following de-compression:
        #        hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
        if self.unpad:
            if self.pad == True and self.enable_stream == False:
                hidden_states = hidden_states.view(maxseqlen, batch, hidden_size).permute(1,0,2).contiguous().view(batch,maxseqlen,hidden_size).contiguous()
            if self.pad == True and self.enable_stream == True:
                hidden_states = hidden_states.view(batch,maxseqlen,hidden_size)
            if self.pad == False:
                hidden_states = PadInput.apply(hidden_states, attention_indices, batch, maxseqlen, hidden_size, ntokens).view(batch, maxseqlen, hidden_size).contiguous()

        if not output_all_encoded_layers or checkpoint_activations:
            if self.fused_mha:
                all_encoder_layers.append(hidden_states.permute(1,0,2).contiguous())
            else:
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def convert_model(model: BertForPreTraining):
    bert_config = copy.copy(model.config)

    bert_config.fused_mha = config.fused_mha
    bert_config.fused_gelu_bias = config.fused_gelu_bias
    bert_config.dense_seq_output = config.dense_seq_output
    bert_config.unpad = config.unpad
    bert_config.unpad_fmha = config.unpad_fmha
    bert_config.pad = config.pad
    bert_config.fuse_qkv = not config.disable_fuse_qkv
    bert_config.fuse_scale = not config.disable_fuse_scale
    bert_config.fuse_mask = not config.disable_fuse_mask
    bert_config.fuse_dropout = config.enable_fuse_dropout
    bert_config.fused_dropout_add = config.fused_dropout_add
    bert_config.apex_softmax = not config.disable_apex_softmax
    bert_config.enable_stream = config.enable_stream
    if bert_config.fuse_mask == True: bert_config.apex_softmax = True
    if bert_config.pad == False: bert_config.enable_stream = True
    if bert_config.unpad == True: bert_config.fused_mha = False

    state_dict = model.state_dict()
    if bert_config.fused_mha:
        state_dict = remap_attn_parameters(state_dict)
    model.bert_model_segment.bert.unpad = bert_config.unpad
    model.bert_model_segment.bert.encoder = NvidiaBertEncoder(bert_config).to(torch.cuda.current_device())
    model.load_state_dict(state_dict, strict=True)
    return model


def remap_attn_parameters(model_dict):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'attention' in k:
            if 'self.query.weight' in k:
                new_k = k.replace('self.query.weight', 'multi_head_attention.q_weight')
            elif 'self.key.weight' in k:
                new_k = k.replace('self.key.weight', 'multi_head_attention.k_weight')
            elif 'self.value.weight' in k:
                new_k = k.replace('self.value.weight', 'multi_head_attention.v_weight')
            elif 'self.query.bias' in k:
                new_k = k.replace('self.query.bias', 'multi_head_attention.q_bias')
            elif 'self.key.bias' in k:
                new_k = k.replace('self.key.bias', 'multi_head_attention.k_bias')
            elif 'self.value.bias' in k:
                new_k = k.replace('self.value.bias', 'multi_head_attention.v_bias')
            elif 'output.dense.weight' in k:
                new_k = k.replace('output.dense.weight', 'multi_head_attention.out_proj_weight')
            elif 'output.dense.bias' in k:
                new_k = k.replace('output.dense.bias', 'multi_head_attention.out_proj_bias')
            elif 'output.LayerNorm.weight' in k:
                new_k = k.replace('output.LayerNorm.weight', 'layer_norm.weight')
            elif 'output.LayerNorm.bias' in k:
                new_k = k.replace('output.LayerNorm.bias', 'layer_norm.bias')
            else:
                new_k = k
        else:
            new_k = k
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Transformer."""
import math
from typing import Optional, List
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm

from model.models.module import MegatronModule
from model.layers import layers
from model.models.enums import AttnMaskType, LayerType, AttnType
from model.layers.fused_softmax import FusedScaleMaskSoftmax
from model.layers.fused_bias_gelu import bias_gelu_impl
from model.layers.utils import attention_mask_func
import config

rearrange = None

flash_attn_unpadded_func = None


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty(
        (1,),
        dtype = inp.dtype,
        device = inp.device,
        requires_grad = requires_grad,
    )
    out.data = inp.data
    return out

class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''
    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output

def _args_to_kwargs():

    common_kwargs = {
        "params_dtype": config.params_dtype,
        "use_cpu_initialization": True,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel_enabled": False,
    }
    return common_kwargs

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = layers.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            **_args_to_kwargs())

        self.bias_gelu_fusion = False
        self.activation_func = None

        self.bias_gelu_fusion = config.bias_gelu_fusion
        self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = layers.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

class CoreAttention(MegatronModule):

    def __init__(self, layer_number,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = False

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            config.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.zeros(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            dtype=query_layer.dtype, device=key_layer.device)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            self.dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            self.dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.sequence_parallel = False

        self.use_flash_attn = config.use_flash_attn \
            and attention_type == AttnType.self_attn \
            and self.attn_mask_type == AttnMaskType.causal
        if self.use_flash_attn:
            if flash_attn_unpadded_func is None:
                raise ImportError('FlashAttention is not installed, please install with '
                                  'pip install flash-attn')
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = layers.ColumnParallelLinear(
                config.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method,
                **_args_to_kwargs())
        else:
            assert attention_type == AttnType.cross_attn
            self.query = layers.ColumnParallelLinear(
                config.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
                **_args_to_kwargs())


            self.key_value = layers.ColumnParallelLinear(
                config.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method,
                **_args_to_kwargs())

        self.core_attention = CoreAttention(self.layer_number,
                                            self.attn_mask_type)

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=config.attention_dropout
            )

        # Output.
        self.dense = layers.RowParallelLinear(
            projection_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)
            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = torch.split(mixed_x_layer, mixed_x_layer.size()[-1] // 3, mixed_x_layer.dim() - 1)

        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = torch.split(mixed_kv_layer, mixed_kv_layer.size()[-1] // 2, mixed_kv_layer.dim() - 1)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]

        # ==================================
        # core attention computation
        # ==================================

        if not self.use_flash_attn:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            if not self.sequence_parallel:
                context_layer = self.core_attention_flash(q, k, v)
            else:
                context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)
        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection

        # Layernorm on the input data.
        # self.input_layernorm = LayerNorm(
        #     config.hidden_size,
        #     eps=config.layernorm_epsilon)
        self.input_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=torch.float)
            # eps=config.layernorm_epsilon,
            # no_persist_layer_norm=config.no_persist_layer_norm,
            # sequence_parallel=config.sequence_parallel)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        # self.post_attention_layernorm = LayerNorm(
        #     config.hidden_size,
        #     eps=config.layernorm_epsilon)
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=torch.float)
            # eps=config.layernorm_epsilon,
            # no_persist_layer_norm=config.no_persist_layer_norm,
            # sequence_parallel=config.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            # self.post_inter_attention_layernorm = LayerNorm(
            #     config.hidden_size,
            #     eps=config.layernorm_epsilon)
            self.post_inter_attention_layernorm = LayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                dtype=torch.float)
                # eps=config.layernorm_epsilon,
                # no_persist_layer_norm=config.no_persist_layer_norm,
                # sequence_parallel=config.sequence_parallel)

        # MLP
        self.mlp = ParallelMLP(init_method, output_layer_init_method)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = make_viewless_tensor(inp = output,
                                                     requires_grad = output.requires_grad,
                                                     keep_graph = True)

        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        return output


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


def _get_num_layers(is_decoder=False):
    """Compute the number of transformer layers resident on the current rank."""
    if not is_decoder:
        num_layers = config.encoder_num_layers
    else:
        num_layers = config.decoder_num_layers
    return num_layers


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True, post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()

        self.layer_type = layer_type
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = config.transformer_impl

        self.sequence_parallel = False

        self.microbatch_count = 0

        # Number of layers.
        self.num_layers = _get_num_layers(
            layer_type == LayerType.decoder)

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, config.num_layers)]

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])

        offset = 0 

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            # self.final_layernorm = LayerNorm(
            #     config.hidden_size,
            #     eps=config.layernorm_epsilon)
            self.final_layernorm = LayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                dtype=torch.float)
                #no_persist_layer_norm=config.no_persist_layer_norm,
                #sequence_parallel=config.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        rng_context = nullcontext()

        with rng_context:
            # The fp8_autocast context manager is a no-op when enabled=True
            # The if...else serves to short circuit name resolution for fp8_autocast
            with nullcontext():

                # Forward pass.
                forward_kwargs = {
                    'encoder_output': encoder_output,
                    'enc_dec_attn_mask': enc_dec_attn_mask,
                    'inference_params': inference_params,
                }

                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(
                        hidden_states,
                        attention_mask,
                        **forward_kwargs)

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    self.microbatch_count += 1

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
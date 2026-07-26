import torch
import torch_xla
import torch.nn.functional as F
import torch
import torch.nn as nn
from tx8_util import IdentityToAllReduce,AllReduceToIdentity
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
import math

def replace_tx8_llama_rotary_embedding():
    import transformers.models.llama.modeling_llama as llama
    if torch_xla._XLAC.IsCurrentDeviceTx8() and hasattr(llama, 'LlamaRotaryEmbedding'):
        from torch import nn
        class TX8RotaryEmbedding(nn.Module):
            def __init__(self,config = None):
                super().__init__()
            @torch.no_grad()
            def forward(self, x, position_ids):
                return None, None
        print('!!!! replace transformers.models.qwen2.modeling_llama.LlamaRotaryEmbedding to', TX8RotaryEmbedding.__name__)
        llama.LlamaRotaryEmbedding = TX8RotaryEmbedding


def replace_tx8_mask():
    from torch_xla.tx8_custom_ops.causal_attention_mask import replace_tx8_causal_attention_mask_func
    replace_tx8_causal_attention_mask_func()

def replace_tx8_rmsNorm(model):
    from torch_xla.tx8_custom_ops.replace_custom_ops import tx8_model_optimize
    model = tx8_model_optimize(model)
    return model

# 使用FlashAttention时，不存在Mask算子
def enable_tx8_flash_attention(model):
    from torch_xla.tx8_custom_ops.attention import enable_flash_attention
    return enable_flash_attention(model)

def replace_TrainingArguments_setup_devices():
    import os
    from datetime import timedelta
    import warnings
    import transformers.training_args as training_args
    from transformers.training_args import (logger,ParallelMode,)
    from transformers.utils import (
        ACCELERATE_MIN_VERSION,
        cached_property,
        is_accelerate_available,
        is_ipex_available,
        is_sagemaker_dp_enabled,
        is_sagemaker_mp_enabled,
        is_torch_mlu_available,
        is_torch_mps_available,
        is_torch_musa_available,
        is_torch_npu_available,
        is_torch_xla_available,
        is_torch_xpu_available,
        requires_backends,
    )
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp

    from transformers.utils.generic import strtobool
    import torch.distributed as dist

    from accelerate.state import AcceleratorState, PartialState
    from accelerate.utils import DistributedType
    from transformers.trainer_pt_utils import AcceleratorConfig
    import torch_xla.core.xla_model as xm

    @cached_property
    def _setup_devices_new(self) -> "torch.device":
        requires_backends(self, ["torch"])
        logger.info("PyTorch: setting up devices")
        if not is_sagemaker_mp_enabled():
            if not is_accelerate_available():
                raise ImportError(
                    f"Using the `Trainer` with `PyTorch` requires `accelerate>={ACCELERATE_MIN_VERSION}`: "
                    "Please run `pip install transformers[torch]` or `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )
        # We delay the init of `PartialState` to the end for clarity
        accelerator_state_kwargs = {"enabled": True, "use_configured_state": False}
        if isinstance(self.accelerator_config, AcceleratorConfig):
            accelerator_state_kwargs["use_configured_state"] = self.accelerator_config.pop(
                "use_configured_state", False
            )
        if accelerator_state_kwargs["use_configured_state"]:
            if PartialState._shared_state == {}:
                raise ValueError(
                    "Passing `'use_configured_state':True` to the AcceleratorConfig requires a pre-configured "
                    "`AcceleratorState` or `PartialState` to be defined before calling `TrainingArguments`. "
                )
            # We rely on `PartialState` to yell if there's issues here (which it will)
            self.distributed_state = PartialState(cpu=self.use_cpu)
            if self.deepspeed and self.distributed_state.distributed_type != DistributedType.DEEPSPEED:
                raise RuntimeError(
                    "Tried to use an already configured `Accelerator` or `PartialState` that was not initialized for DeepSpeed, "
                    "but also passed in a `deepspeed` configuration to the `TrainingArguments`. Please set "
                    "`use_configured_state:False` instead or setup your `Accelerator` or `PartialState` properly."
                )
        else:
            AcceleratorState._reset_state(reset_partial_state=True)
            self.distributed_state = None
        if not self.use_ipex and "ACCELERATE_USE_IPEX" not in os.environ:
            os.environ["ACCELERATE_USE_IPEX"] = "false"

        self._n_gpu = 1
        if self.use_cpu or strtobool(os.environ.get("ACCELERATE_USE_CPU", "False")):
            accelerator_state_kwargs["cpu"] = True
            accelerator_state_kwargs["backend"] = self.ddp_backend
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            accelerator_state_kwargs["enabled"] = False
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        elif is_sagemaker_dp_enabled():
            accelerator_state_kwargs["_use_sagemaker_dp"] = True
        elif self.deepspeed:
            accelerator_state_kwargs["use_deepspeed"] = True
            accelerator_state_kwargs["timeout"] = timedelta(seconds=self.ddp_timeout)
        else:
            accelerator_state_kwargs["backend"] = self.ddp_backend
            accelerator_state_kwargs["timeout"] = timedelta(seconds=self.ddp_timeout)

        # Now we pop everything
        if accelerator_state_kwargs.pop("enabled", False) and not accelerator_state_kwargs.pop(
            "use_configured_state", False
        ):
            # We need to patch this env var when enabling to detect deepspeed
            use_deepspeed = accelerator_state_kwargs.pop("use_deepspeed", False)
            if use_deepspeed:
                os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(**accelerator_state_kwargs)
            if use_deepspeed:
                del os.environ["ACCELERATE_USE_DEEPSPEED"]
        if not is_sagemaker_mp_enabled():
            device = self.distributed_state.device
            self.local_rank = self.distributed_state.local_process_index
        if dist.is_available() and dist.is_initialized() and self.parallel_mode != ParallelMode.DISTRIBUTED:
            logger.warning(
                "torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        if is_torch_xla_available():
            #device = self.distributed_state.device
            device = xm.xla_device()
            self._n_gpu = 0
        elif is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled():
            # Already set _n_gpu
            pass
        elif self.distributed_state.distributed_type == DistributedType.NO:
            if self.use_mps_device:
                warnings.warn(
                    "`use_mps_device` is deprecated and will be removed in version 5.0 of 🤗 Transformers. "
                    "`mps` device will be used by default if available similar to the way `cuda` device is used."
                    "Therefore, no action from user is required. "
                )
                if device.type != "mps":
                    raise ValueError(
                        "Either you do not have an MPS-enabled device on this machine or MacOS version is not 12.3+ "
                        "or current PyTorch install was not built with MPS enabled."
                    )
            if self.use_cpu:
                device = torch.device("cpu")
            elif is_torch_mps_available():
                device = torch.device("mps")
            elif is_torch_xpu_available():
                if not is_ipex_available() and not is_accelerate_available("0.32.0.dev"):
                    raise ImportError("Using the XPU PyTorch backend requires `accelerate>=0.32.0.dev`")
                device = torch.device("xpu:0")
                torch.xpu.set_device(device)
            elif is_torch_mlu_available():
                device = torch.device("mlu:0")
                torch.mlu.set_device(device)
            elif is_torch_musa_available():
                device = torch.device("musa:0")
                torch.musa.set_device(device)
            elif is_torch_npu_available():
                device = torch.device("npu:0")
                torch.npu.set_device(device)
            else:
                # if n_gpu is > 1 we'll use nn.DataParallel.
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
                # trigger an error that a device index is missing. Index 0 takes into account the
                # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
                # will use the first GPU in that env, i.e. GPU#1
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else os.environ.get("ACCELERATE_TORCH_DEVICE", "cpu")
                )
                # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
                # the default value.
                self._n_gpu = torch.cuda.device_count()
                if device.type == "cuda":
                    torch.cuda.set_device(device)
        return device

    print('!!!! replace transformers.training_args.TrainingArguments._setup_devices to _setup_devices_new')
    training_args.TrainingArguments._setup_devices = _setup_devices_new


def replace_coreattention(model):
    for name, module in model.named_children():
        if module.__class__.__name__ == "CoreAttention":
            new_module = TX8CoreAttention(module)
            setattr(model, name, new_module)
        else:
            replace_coreattention(module)
            
    return model

def replace_customcall(model):
    for name, module in model.named_children():
        if module.__class__.__name__ == "RMSNorm": 
            # Assume the new module can be instantiated without arguments
            # print(*module.parameters())
            new_module = RMSNorm(module)
            setattr(model, name, new_module)
        elif module.__class__.__name__ == "Embedding": 
            new_module = Embedding(module)
            setattr(model, name, new_module)
        else:
            replace_customcall(module)

    return model

def replace_unsupportcall(model):
    from types import MethodType
    if model.__class__.__name__ == "ChatGLMForConditionalGeneration": 
        ConditionalGeneration_new_forward = MethodType(ConditionalGeneration_forward, model)
        model.forward = ConditionalGeneration_new_forward
    for name, module in model.named_children():
        if module.__class__.__name__ == "ChatGLMModel": 
            ChatGLMModel_new_forward = MethodType(ChatGLMModel_forward, module)
            module.forward = ChatGLMModel_new_forward

    return model


def replace_rope_embedding(model):
    for name, module in model.named_children():
        if module.__class__.__name__ == "RotaryEmbedding":
            new_module = RotaryEmbedding(module)
            setattr(model, name, new_module)
        else:
            replace_rope_embedding(module)
            
    return model

def replace_self_attention_forward(model):
    from types import MethodType
    for name, module in model.named_children():
        if module.__class__.__name__ == "SelfAttention":
            rope_forward_new_impl = MethodType(self_attention_forward, module)
            module.forward = rope_forward_new_impl
        else:
            replace_self_attention_forward(module)
            
    return model


class Embedding(torch.nn.Module):
    """Language model embeddings."""
    def __init__(self, module):
        super(Embedding, self).__init__()

        self.hidden_size = module.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = module.word_embeddings
        self.fp32_residual_connection = module.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        # embeddings = embeddings.transpose(0, 1).contiguous() # remove
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings

class RMSNorm(nn.Module):
    def __init__(self, original_module):
        super(RMSNorm, self).__init__()
        self.weight = original_module.weight
        self.variance_epsilon = original_module.eps
        self.type = 1

    def forward(self, x):
        return torch.ops.tx8_ops.Rmsnorm(x, self.weight, self.variance_epsilon, self.type)

class RotaryEmbedding(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.dim = original_module.dim
        self.device = "cpu"
        self.dtype = original_module.inv_freq.dtype

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, position_ids, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        if not hasattr(self, "cache"):
            theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))
            # Create position indexes `[0, 1, ..., seq_len - 1]`
            seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

            # Calculate the product of position index and $\theta_i$
            idx_theta = torch.outer(seq_idx, theta).float()

            cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

            # this is to mimic the behaviour of complex32, else we will get different results
            if dtype in (torch.float16, torch.bfloat16, torch.int8):
                cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
            if position_ids is not None: # train scene, position_ids must be the same ,then use cache
                position_ids = position_ids.to(self.device).view(-1) 
                cache = torch.index_select(cache, 0, position_ids)
                cache = cache.unsqueeze(0)
            else:
                cache = cache[None, :seq_len] # after padding max_seq_len , seq_length = max_seq_len
            self.cache = cache
            
        return self.cache

    def forward(self, max_seq_len, position_ids, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.dtype, device= self.device , position_ids = position_ids 
        )


class TX8CoreAttention(nn.Module):
    def __init__(self, module):
        super(TX8CoreAttention, self).__init__()
        self.apply_query_key_layer_scaling = module.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = module.attention_softmax_in_fp32
        self.layer_number = module.layer_number
        self.hidden_size_per_partition = module.hidden_size_per_partition
        self.hidden_size_per_attention_head = module.hidden_size_per_attention_head
        self.num_attention_heads_per_partition = module.num_attention_heads_per_partition
        self.norm_factor = module.norm_factor
        self.coeff = module.coeff
        self.attention_dropout = module.attention_dropout
    
    def tx8_to_4d_for_transformer(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        start_pos = key_value_length - query_length
        batch_size = attention_mask_2d.shape[0]
        from torch_xla.tx8_custom_ops.causal_attention_mask import CausalAttentionMask
        causal_mask = CausalAttentionMask.apply(start_pos, query_length)
        if causal_mask.dtype is not dtype:
            causal_mask = causal_mask.to(dtype)

        if batch_size > 1:
            return causal_mask.expand(batch_size, 1, query_length, query_length + start_pos)
        else:
            return causal_mask
        
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # 获取 dtype 和设备信息
        dtype = query_layer.dtype  # 假设 dtype 和 query_layer 一致
        device = query_layer.device
        query_length = query_layer.size(1)  # 查询序列的长度
        key_value_length = key_layer.size(1)  # 键/值的长度（可能不同）

        # 使用 to_4d 函数处理 attention_mask（将其转换为 4D 掩码）
        attention_mask_4d = self.tx8_to_4d_for_transformer(
            attention_mask_2d=attention_mask,
            query_length=query_length,
            dtype=dtype,
            key_value_length=key_value_length
        )

        # 定义输出的尺寸 [B, NH, SqL, CaSqL]
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        Use_FlashAttn = True
        if Use_FlashAttn :
            from torch_xla.tx8_custom_ops.attention import AttentionOp
            start_pos = torch.tensor([0], dtype=torch.int32).to(key_layer.device)
            seq_len = torch.tensor([key_layer.shape[2]], dtype = torch.int32).to(key_layer.device)
            context_layer, _ = AttentionOp.apply(query_layer, key_layer, value_layer, 
                                                torch.Tensor([ math.sqrt(query_layer.size(-1)) ]).to(torch.float32).to(key_layer.device), 
                                                start_pos, seq_len) # pytorch >= 2.0 
        else:
            matmul_result = torch.matmul(query_layer, key_layer.transpose(2, 3)) / self.norm_factor
            # [ B , NH, Sq, CaSql]
            attention_scores = matmul_result

            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()  

            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff

            attention_probs = F.softmax(attention_scores + attention_mask_4d, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            attention_probs = self.attention_dropout(attention_probs)
            # [ B , NH, Sq, CaSql]
            
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose(1,2).contiguous() # [B, Sq, NH, H]
        # 最后重塑为每个分区的隐藏层大小
        # new_context_layer_shape = (context_layer.size(0), context_layer.size(1), self.hidden_size_per_partition)
        context_layer = context_layer.view(context_layer.size(0), context_layer.size(1), self.hidden_size_per_partition)

        return context_layer
        
def apply_rotary_pos_emb_tx8(q_x: torch.Tensor, k_x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # 获取输入 tensor 的形状
    sq, b, np, hn = q_x.size(0), q_x.size(1), q_x.size(2), q_x.size(3)
    # 获取 rope_cache 的维度，进行调整
    rot_dim = rope_cache.shape[-2] * 2
    
    def pre_process(x):
        np = x.size(2)
        # 进行切分，保留要进行位置编码的部分
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]

        # xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        # x = torch.cat((xshaped[..., 0], xshaped[..., 1]), dim=-1)
        x = x.permute(0, 2, 1, 3)
        return x, x_pass
    
    # rope_cache = rope_cache[:sq]
    # cos = torch.cat((rope_cache[..., 0], rope_cache[..., 0]), dim=-1)
    # sin = torch.cat((rope_cache[..., 1], rope_cache[..., 1]), dim=-1)
    if not hasattr(apply_rotary_pos_emb_tx8, "cos") and not hasattr(apply_rotary_pos_emb_tx8, "sin"):
        rope_cache = rope_cache[:sq]
        apply_rotary_pos_emb_tx8.cos = torch.cat((rope_cache[..., 0], rope_cache[..., 0]), dim=-1).squeeze(0)
        apply_rotary_pos_emb_tx8.sin = torch.cat((rope_cache[..., 1], rope_cache[..., 1]), dim=-1).squeeze(0)
    q_x, q_x_pass = pre_process(q_x)
    k_x, k_x_pass = pre_process(k_x)
    from torch_xla.tx8_custom_ops.rope import tx8_rope
    q_embed, k_embed = tx8_rope(q_x, k_x, apply_rotary_pos_emb_tx8.cos.to(q_x.device), apply_rotary_pos_emb_tx8.sin.to(q_x.device))
    
    def end_tx8_rope(x, q_x_pass):
        x = x.permute(0, 2, 1, 3)
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # q_embed = torch.stack([x1, x2], -1)
        # q_embed = q_embed.flatten(3)
        
        # 将旋转后的部分与未旋转的部分拼接
        x_out2 = torch.cat((x, q_x_pass), dim=-1)
        return x_out2
    return end_tx8_rope(q_embed, q_x_pass), end_tx8_rope(k_embed, k_x_pass)
   

def self_attention_forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================

    # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                            3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        # query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        # key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
        query_layer, key_layer = apply_rotary_pos_emb_tx8(query_layer, key_layer, rotary_pos_emb)

    # adjust key and value for inference
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        key_layer = torch.cat((cache_k, key_layer), dim=0)
        value_layer = torch.cat((cache_v, value_layer), dim=0)
    if use_cache:
        kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(-2)
        key_layer = key_layer.expand(
            -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.unsqueeze(-2)
        value_layer = value_layer.expand(
            -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )

    # ==================================
    # core attention computation
    # ==================================

    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

    # =================
    # Output. [sq, b, h]
    # =================

    output = self.dense(context_layer)

    return output, kv_cache

def ChatGLMModel_forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                dtype=inputs_embeds.dtype)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                        attention_mask], dim=-1)

    if full_attention_mask is None:
        if (attention_mask is not None) or (past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length, position_ids)
    # if position_ids is not None:
    #     position_ids = position_ids.view(-1)
    #     rotary_pos_emb = torch.index_select(rotary_pos_emb, 0, position_ids)
    #     rotary_pos_emb = rotary_pos_emb.unsqueeze(0)
    # else:
    #     rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    # rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous() # [Bs, Seq ,h]

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )
    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )

def ConditionalGeneration_forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        # lm_logits = lm_logits.transpose(0, 1).contiguous() #just remove here

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

def gen_new_weight_order(config):
    """
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]

        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        x = torch.cat((xshaped[..., 0], xshaped[..., 1]), dim=-1)
    """
    projection_size = config.kv_channels * config.num_attention_heads
    hidden_size_per_attention_head = projection_size // config.num_attention_heads

    #  64 rotary_dim
    rotary_dim = (config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels) // 2 

    # q_end = 4096 projection_size
    q_end = projection_size

    # k_end = 4096 + 256 
    k_end = q_end + hidden_size_per_attention_head * config.multi_query_group_num if config.multi_query_attention else projection_size

    # v_end = k_end + 256
    v_end = k_end + hidden_size_per_attention_head * config.multi_query_group_num if config.multi_query_attention else projection_size

    device = 'cpu'
    new_order = []
    for block_start in range(0, q_end, hidden_size_per_attention_head):
        block_indices = torch.arange(block_start, block_start + rotary_dim, device = device) 
        even_indices = block_indices[::2] 
        odd_indices = block_indices[1::2]  
        new_order += even_indices.tolist() + odd_indices.tolist()
        block_indices = torch.arange(block_start + rotary_dim, block_start + 2 * rotary_dim, device = device) 
        new_order += block_indices.tolist()
 
    for block_start in range(q_end, k_end, hidden_size_per_attention_head):
        block_indices = torch.arange(block_start, block_start + rotary_dim, device = device) 
        even_indices = block_indices[::2]  
        odd_indices = block_indices[1::2]  
        new_order += even_indices.tolist() + odd_indices.tolist()

        block_indices = torch.arange(block_start + rotary_dim, block_start + 2 * rotary_dim, device = device) 
        new_order += block_indices.tolist()

    block_indices = torch.arange(k_end, v_end, device = device)
    new_order += block_indices.tolist()
    return new_order


def gen_origin_order(config):
    """
        return origin weigth order, prepare for save origin weight order
    """
    device = 'cpu'
    projection_size = config.kv_channels * config.num_attention_heads
    hidden_size_per_attention_head = projection_size // config.num_attention_heads
    #  64 rotary_dim
    rotary_dim = (config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels) // 2 
    # q_end = projection_size
    q_end = projection_size
    k_end = q_end + hidden_size_per_attention_head * config.multi_query_group_num if config.multi_query_attention else projection_size
    v_end = k_end + hidden_size_per_attention_head * config.multi_query_group_num if config.multi_query_attention else projection_size
    # q
    new_order = []
    for block_start in range(0, q_end, hidden_size_per_attention_head):

        front_indices = torch.arange(block_start, block_start + rotary_dim // 2, device = device)
        back_indices = torch.arange(block_start + rotary_dim // 2, block_start + rotary_dim, device = device)

        interleaved_indices = torch.stack([front_indices, back_indices], dim=1).view(-1)
        new_order += interleaved_indices.tolist()
        block_indices = torch.arange(block_start + rotary_dim, block_start + 2 * rotary_dim, device = device) #
        new_order += block_indices.tolist()
    # k
    for block_start in range(q_end, k_end, hidden_size_per_attention_head):

        front_indices = torch.arange(block_start, block_start + rotary_dim // 2)
        back_indices = torch.arange(block_start + rotary_dim // 2, block_start + rotary_dim, device = device)

        interleaved_indices = torch.stack([front_indices, back_indices], dim=1).view(-1)
        new_order += interleaved_indices.tolist()
        block_indices = torch.arange(block_start + rotary_dim, block_start + 2 * rotary_dim, device = device) # 
        new_order += block_indices.tolist()
    # v
    block_indices = torch.arange(k_end, v_end, device = device)
    new_order += block_indices.tolist()
    return new_order
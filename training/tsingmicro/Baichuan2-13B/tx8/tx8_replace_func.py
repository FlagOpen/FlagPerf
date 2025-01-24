import math
import torch
import torch_xla
from tx8_util import IdentityToAllReduce,AllReduceToIdentity
from typing import List, Optional, Tuple, Union
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.get_logger(__name__)

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

def decode_layer_forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,  # will become mandatory in v4.46
        **kwargs,
    ) :
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention   
        hidden_states = IdentityToAllReduce.apply(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = AllReduceToIdentity.apply(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = IdentityToAllReduce.apply(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = AllReduceToIdentity.apply(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

def replace_tx8_llama_decode_forward():
    import transformers.models.llama.modeling_llama as llama
    if torch_xla._XLAC.IsCurrentDeviceTx8() and hasattr(llama, 'LlamaDecoderLayer'):
        layer = getattr(llama, 'LlamaDecoderLayer')
        layer.forward = decode_layer_forward
        print('!!!! replace transformers.models.qwen2.modeling_llama.LlamaMLP.forward to decode_layer_forward')

def precompute_sin_cos(config):
    max_position_embeddings = config.max_position_embeddings
    position_ids = torch.arange(max_position_embeddings).unsqueeze(0)
    dtype = torch.bfloat16
    dim = config.hidden_size // config.num_attention_heads
    base = config.rope_theta

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype).squeeze(0) # 这里删除 sin和cos的 batch_size 的维度.
    sin = emb.sin().to(dtype=dtype).squeeze(0)
    return sin, cos

def replace_tx8_rope(config):
    from torch_xla.tx8_custom_ops.rope import tx8_rope, replace_tx8_rope_func_with_func
    sin_cpu, cos_cpu = precompute_sin_cos(config)
    def tx8_rope_sin_cos(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
        q, k = tx8_rope(q, k, cos_cpu.to(q.device), sin_cpu.to(k.device))
        return q, k
    replace_tx8_rope_func_with_func(tx8_rope_sin_cos)   

def replace_tx8_mask():
    from torch_xla.tx8_custom_ops.causal_attention_mask import replace_tx8_causal_attention_mask_func
    replace_tx8_causal_attention_mask_func()

def replace_tx8_rmsNorm(model):
    from torch_xla.tx8_custom_ops.replace_custom_ops import tx8_model_optimize
    model = tx8_model_optimize(model)
    return model

class TX8FlashAttention(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        config = module.config
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = module.W_pack
        self.o_proj = module.o_proj
        # self.W_pack = torch.nn.Linear(
        #     self.hidden_size, 3 * self.hidden_size, bias=False
        # )
        # self.o_proj = torch.nn.Linear(
        #     self.num_heads * self.head_dim, self.hidden_size, bias=False
        # )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        xops = None
        if xops is not None and self.training:
            attn_weights = None
            # query_states = query_states.transpose(1, 2)
            # key_states = key_states.transpose(1, 2)
            # value_states = value_states.transpose(1, 2)
            # attn_output = xops.memory_efficient_attention(
            #     query_states, key_states, value_states, attn_bias=attention_mask
            # )
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask)
            attn_output = attn_output.transpose(1, 2)
        else:
            from torch_xla.tx8_custom_ops.attention import AttentionOp
            start_pos = torch.tensor([0], dtype=torch.int32).to(key_states.device)
            seq_len = torch.tensor([key_states.shape[2]], dtype = torch.int32).to(key_states.device)
            attn_output, _ = AttentionOp.apply(query_states, key_states, value_states, torch.Tensor([math.sqrt(self.head_dim)]).to(torch.float32).to(key_states.device), start_pos, seq_len)
            # attn_weights = torch.matmul(
            #     query_states, key_states.transpose(2, 3)
            # ) / math.sqrt(self.head_dim)
            # if attention_mask is not None:
            #     if q_len == 1:  # inference with cache
            #         if len(attention_mask.size()) == 4:
            #             attention_mask = attention_mask[:, :, -1:, :]
            #         else:
            #             attention_mask = attention_mask[:, -1:, :]
            #     attn_weights = attn_weights + attention_mask
            #     attn_weights = torch.max(
            #         attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            #     )

            # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            # attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def replace_attention(model):
    for name, module in model.named_children():
        if module.__class__.__name__ == "BaichuanAttention":
            new_module = TX8FlashAttention(module)
            setattr(model, name, new_module)
            print('!!!! replace_tx8_flash_attention SUCCESS')
        else:
            replace_attention(module)
            
    return model

# 使用FlashAttention时，不存在Mask算子
def enable_tx8_flash_attention(model):
    return replace_attention(model)

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

def replace_torch_tril_gt_ge_eq_ne_lt_le():
    _original_tril = torch.tril
    def float_tril(input, diagonal=0):
        result = _original_tril(input,diagonal)
        return result.to(torch.float)
    torch.tril = float_tril

    _original_gt = torch.gt
    def float_gt(input, diagonal=0):
        result =  _original_gt(input, diagonal)
        return result.to(torch.float)
    torch.gt = float_gt

    _original_ge = torch.ge
    def float_ge(input, diagonal=0):
        result =  _original_ge(input, diagonal)
        return result.to(torch.float)
    torch.ge = float_ge

    _original_eq = torch.eq
    def float_eq(input, diagonal=0):
        result = _original_eq(input, diagonal)
        return result.to(torch.float)
    torch.eq = float_eq

    _original_ne = torch.ne
    def float_ne(input, diagonal=0):
        result = _original_ne(input, diagonal)
        return result.to(torch.float)
    torch.ne = float_ne

    _original_lt = torch.lt
    def float_lt(input, diagonal=0):
        result =  _original_lt(input, diagonal)
        return result.to(torch.float)
    torch.lt = float_lt

    _original_le = torch.le
    def float_le(input, diagonal=0):
        result =  _original_le(input, diagonal)
        return result.to(torch.float)
    torch.le = float_le
    print('!!!! replace torch_tril_gt_ge_eq_ne_lt_le to f32')

def new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    proj = self.W_pack(hidden_states)
    proj = (
        proj.unflatten(-1, (3, self.hidden_size))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
    )
    query_states = (
        proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    key_states = (
        proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    value_states = (
        proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    xops = None
    if xops is not None and self.training:
        attn_weights = None
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)
        # attn_output = xops.memory_efficient_attention(
        #     query_states, key_states, value_states, attn_bias=attention_mask
        # )
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask)
        attn_output = attn_output.transpose(1, 2)
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, dtype=torch.bfloat16)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def replace_BaichuanAttention_forward(model):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "BaichuanAttention":
            setattr(module, "forward", new_forward.__get__(module, module.__class__))
            print('!!!! replace BaichuanAttention.forward to new_forward SUCCESS')
        
    return model

def BaichuanModel_new_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot provide both input_ids and inputs_embeds simultaneously"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You need to provide input_ids or inputs_embeds")

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    self.gradient_checkpointing = False

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    seq_length_with_past = seq_length

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self.training:
        if (
            self.alibi_mask is None
            or self.alibi_mask.shape[-1] != seq_length_with_past
        ):
            self.alibi_mask = self.get_alibi_mask(
                inputs_embeds, seq_length_with_past
            )
        alibi_mask = self.alibi_mask


    else:
        alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

    if attention_mask is not None:
        if len(attention_mask.shape) == 2:
            expanded_mask = attention_mask.to(alibi_mask.dtype)
            expanded_mask = torch.tril(
                torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
            ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
        else:
            expanded_mask = attention_mask
        bsz = inputs_embeds.size(0)
        src_len, tgt_len = alibi_mask.size()[-2:]
        expanded_mask = (
            expanded_mask.unsqueeze(1)
            .expand(bsz, 1, src_len, tgt_len)
            .to(alibi_mask.dtype)
        )
        inverted_mask = 1.0 - expanded_mask
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min
        )
        attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
    else:
        attention_mask = alibi_mask

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = (
            past_key_values[idx] if past_key_values is not None else None
        )

        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def replace_BaichuanModel_forward(model):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "BaichuanModel":
            setattr(module, "forward", BaichuanModel_new_forward.__get__(module, module.__class__))
            print('!!!! replace BaichuanModel.forward to new_forward SUCCESS')
        
    return model




import torch
import torch_xla
from tx8_util import IdentityToAllReduce,AllReduceToIdentity
from typing import List, Optional, Tuple, Union
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP, MixtralSparseMoeBlock
import torch.nn.functional as F
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from torch import nn
from collections import OrderedDict
from transformers.activations import ACT2FN
import copy

def replace_tx8_mixtral_rotary_embedding():
    import transformers.models.mixtral.modeling_mixtral as mixtral
    if torch_xla._XLAC.IsCurrentDeviceTx8() and hasattr(mixtral, 'MixtralRotaryEmbedding'):
        from torch import nn
        class TX8RotaryEmbedding(nn.Module):
            def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
                super().__init__()

            def forward(self, x, seq_len=None):
                return None, None
        print('!!!! replace transformers.models.mixtral.modeling_mixtral.mixtralRotaryEmbedding to', TX8RotaryEmbedding.__name__)
        mixtral.MixtralRotaryEmbedding = TX8RotaryEmbedding

def decode_layer_forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        output_router_logits = False,
        use_cache = False,
        cache_position = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = IdentityToAllReduce.apply(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = AllReduceToIdentity.apply(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = IdentityToAllReduce.apply(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = AllReduceToIdentity.apply(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

def replace_tx8_mixtral_decode_forward():
    import transformers.models.mixtral.modeling_mixtral as mixtral
    if torch_xla._XLAC.IsCurrentDeviceTx8() and hasattr(mixtral, 'MixtralDecoderLayer'):
        layer = getattr(mixtral, 'MixtralDecoderLayer')
        layer.forward = decode_layer_forward
        print('!!!! replace transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer.forward to decode_layer_forward')

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
    replace_mixtral_tx8_rope_func_with_func(tx8_rope_sin_cos)   

def replace_mixtral_tx8_rope_func_with_func(new_func):  
    import transformers.models.mixtral.modeling_mixtral as mixtral
    if torch_xla._XLAC.IsCurrentDeviceTx8() and hasattr(mixtral, 'apply_rotary_pos_emb'):
        print('!!!! replace transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb to', new_func.__name__)
        mixtral.apply_rotary_pos_emb = new_func
    else:
        print('!!!! can not find transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb')
    return

def replace_tx8_mask():
    replace_tx8_causal_attention_mask_func()

def replace_tx8_rmsNorm(model):
    for name, module in model.named_children():
        if module.__class__.__name__ == "MixtralRMSNorm": 
            new_module = RMSNorm(module)
            setattr(model, name, new_module)
        else:
            replace_tx8_rmsNorm(module)
    return model

class RMSNorm(nn.Module):
    def __init__(self, original_module):
        super(RMSNorm, self).__init__()
        self.weight = original_module.weight
        self.variance_epsilon =original_module.variance_epsilon
        self.type = 1

    def forward(self, x):
        # 自定义的前向传播逻辑
        return torch.ops.tx8_ops.Rmsnorm(x, self.weight, self.variance_epsilon, self.type) 


class CausalAttentionMask(torch.autograd.Function):
    max_seq_len = 4096

    @staticmethod
    def load(max_seq_len):
        CausalAttentionMask.max_seq_len = max_seq_len

    @staticmethod
    def forward(start_pos, seq_len):
        if torch_xla._XLAC.IsCurrentDeviceTx8():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            start_pos_tensor = torch.tensor([start_pos], dtype = torch.int32).to(device)
            seq_len_tensor = torch.tensor([seq_len], dtype = torch.int32).to(device)
            causal_mask = torch.ops.tx8_ops.CausalAttentionMask(start_pos_tensor, seq_len_tensor, start_pos, seq_len,
                                                                CausalAttentionMask.max_seq_len)
        else:
            type_min = torch.finfo(torch.float32).min
            causal_mask = torch.full((seq_len, seq_len), type_min)
            causal_mask = torch.triu(causal_mask, diagonal = 1)
            if start_pos > 0:
                causal_mask = torch.cat([torch.zeros(seq_len, start_pos), causal_mask], dim = -1)
            causal_mask = causal_mask[None, None, :, :]

        return causal_mask

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def tx8_to_causal_4d_for_transformer(
    self,
    batch_size: int,
    query_length: int,
    key_value_length: int,
    dtype: torch.dtype,
    device: Union[torch.device, "str"] = "cpu",
) -> Optional[torch.Tensor]:
    print('!!! tx8_to_causal_4d_for_transformer')
    start_pos = key_value_length - query_length
    causal_mask = CausalAttentionMask.apply(start_pos, query_length)
    if causal_mask.dtype is not dtype:
        causal_mask = causal_mask.to(dtype)

    if batch_size > 1:
        return causal_mask.expand(batch_size, 1, query_length, query_length + start_pos)
    else:
        return causal_mask

def tx8_to_4d_for_transformer(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
    start_pos = key_value_length - query_length
    batch_size = attention_mask_2d.shape[0]
    causal_mask = CausalAttentionMask.apply(start_pos, query_length)
    if causal_mask.dtype is not dtype:
        causal_mask = causal_mask.to(dtype)

    if batch_size > 1:
        return causal_mask.expand(batch_size, 1, query_length, query_length + start_pos)
    else:
        return causal_mask

# 替换transformers 4.46.1版本中 llama_model 的 _update_causal_mask 方法
def tx8_update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values, #: Cache,
    output_attentions: bool,
) -> Optional[torch.Tensor]:
    start_pos = past_key_values.get_seq_length() if past_key_values is not None else 0
    sequence_length = cache_position.shape[0]
    causal_mask = CausalAttentionMask.apply(start_pos, sequence_length)
    dtype = input_tensor.dtype
    batch_size=input_tensor.shape[0]
    if causal_mask.dtype is not dtype:
       causal_mask = causal_mask.to(dtype)

    if batch_size > 1:
        return causal_mask.expand(batch_size, 1, sequence_length, sequence_length + start_pos)
    else:
        return causal_mask

def replace_tx8_causal_attention_mask_func():
    import transformers.modeling_attn_mask_utils as tma
    from types import MethodType

    to_causal_4d = MethodType(tx8_to_causal_4d_for_transformer, tma.AttentionMaskConverter)
    tma.AttentionMaskConverter.to_causal_4d = to_causal_4d
    print('!!!! replace transformers.AttentionMaskConverter.to_causal_4d to tx8_to_causal_4d_for_transformer')

    to_4d = MethodType(tx8_to_4d_for_transformer, tma.AttentionMaskConverter)
    tma.AttentionMaskConverter.to_4d = to_4d
    print('!!!! replace transformers.AttentionMaskConverter.to_4d to tx8_to_4d_for_transformer')

    import pkg_resources
    transformers_version = pkg_resources.get_distribution('transformers').version
    if transformers_version == '4.46.1':
      import transformers.models.llama.modeling_llama as modeling_llama
      update_causal_mask = MethodType(tx8_update_causal_mask, modeling_llama.LlamaModel)
      modeling_llama.LlamaModel._update_causal_mask = update_causal_mask
      # modeling_llama.LlamaModel._update_causal_mask = tx8_update_causal_mask
      print('!!!! replace transformers.modeling_llama.LlamaModel._update_causal_mask to tx8_update_causal_mask')

      import transformers.models.mixtral.modeling_mixtral as modeling_mixtral
      update_causal_mask = MethodType(tx8_update_causal_mask, modeling_mixtral.MixtralModel)
      modeling_mixtral.MixtralModel._update_causal_mask = update_causal_mask
      # modeling_llama.LlamaModel._update_causal_mask = tx8_update_causal_mask
      print('!!!! replace transformers.modeling_mixtral.MixtralModel._update_causal_mask to tx8_update_causal_mask')

    return


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

""" 替换torch.nn.functional.one_hot产生的compare <i1>"""
def tx8_one_hot_opt(indices, num_classes = -1, dynamic_shape = False):
    device = indices.device
    if num_classes == -1:
        num_classes = (torch.max(indices) + 1)
    if len(indices.shape) == 1:
        # Step 1: 生成所有类别的索引
        arange_tensor = torch.arange(num_classes, dtype=torch.float32, device=device).unsqueeze(0)  # 
        # Step 2: 将输入张量重复以匹配类别索引的形状
        input_repeated = indices.unsqueeze(1).repeat(1, num_classes)  # 
        arange_repeated = arange_tensor.repeat(indices.size(0), 1)  # 
        # Step 3: 计算差值，得到差矩阵
        diff_matrix = input_repeated - arange_repeated  # 差值比1大
        
        # Step 4: 取绝对值
        abs_matrix = diff_matrix.abs()  # 形状 (batch_size, num_classes)
        # Step 5: 取符号
        # sign_matrix = abs_matrix.sign()  # 如果差值为0，sign为0，否则为1 # 出现compare
        sign_matrix = torch.clamp(abs_matrix, max=1)

        # Step 6: 最终结果为 1 - sign_matrix
        one_hot_matrix = 1 - sign_matrix  # 形状 (batch_size, num_classes)
    elif len(indices.shape) == 2:
        H, W = indices.shape
        input_flat = indices.view(-1)
        # Step 1: 生成所有类别的索引
        arange_tensor = torch.arange(num_classes, dtype=torch.float32, device=device).unsqueeze(0)  # 
    
        # Step 2: 将输入张量重复以匹配类别索引的形状
        input_repeated = input_flat.unsqueeze(1).repeat(1, num_classes)  # 
        arange_repeated = arange_tensor.repeat(input_flat.size(0), 1)  # 
        
        # Step 3: 计算差值，得到差矩阵
        diff_matrix = input_repeated - arange_repeated  # 差值>=1大
        
        # Step 4: 取绝对值
        abs_matrix = diff_matrix.abs()  # 形状 (batch_size, num_classes)
        # Step 5: 取符号
        # sign_matrix = abs_matrix.sign()  # 如果差值为0，sign为0，否则为1 # 出现compare
        if dynamic_shape:
            sign_matrix = torch.clamp(abs_matrix, max=0,min=0)
        else:
            sign_matrix = torch.clamp(abs_matrix, max=1)
        # Step 6: 最终结果为 1 - sign_matrix
        one_hot_matrix = 1 - sign_matrix  # 形状 (batch_size, num_classes)

        one_hot_matrix = one_hot_matrix.view(H, W, num_classes) #
    elif len(indices.shape) == 3:
        C, H, W = indices.shape
        input_flat = indices.view(-1)
        # print('!!!33333!!!!!!')
        # Step 1: 生成所有类别的索引
        arange_tensor = torch.arange(num_classes, dtype=torch.float32, device=device).unsqueeze(0)  # 
    
        # Step 2: 将输入张量重复以匹配类别索引的形状
        input_repeated = input_flat.unsqueeze(1).repeat(1, num_classes)  # 
        arange_repeated = arange_tensor.repeat(input_flat.size(0), 1)  # 
        
        # Step 3: 计算差值，得到差矩阵
        diff_matrix = input_repeated - arange_repeated  # 差值>=1大
        
        # Step 4: 取绝对值
        abs_matrix = diff_matrix.abs()  # 形状 (batch_size, num_classes)
        # Step 5: 取符号
        # sign_matrix = abs_matrix.sign()  # 如果差值为0，sign为0，否则为1 # 出现compare
        sign_matrix = torch.clamp(abs_matrix, max=1)

        # Step 6: 最终结果为 1 - sign_matrix
        one_hot_matrix = 1 - sign_matrix  # 形状 (batch_size, num_classes)

        one_hot_matrix = one_hot_matrix.view(C, H, W, num_classes) #
    
    else:
        raise ValueError("Unsupported tensor shape.")
    return one_hot_matrix


def replace_tx8_one_hot():
    def new_load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
        num_experts: Optional[int] = None,
        top_k=2,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, int]:
        r"""
        Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

        See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
        function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
        experts is too unbalanced.

        Args:
            gate_logits:
                Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
                shape [batch_size X sequence_length, num_experts].
            num_experts:
                Number of experts
            top_k:
                The number of experts to route per-token, can be also interpreted as the `top-k` routing
                parameter.
            attention_mask (`torch.Tensor`, *optional*):
                The attention_mask used in forward function
                shape [batch_size X sequence_length] if not None.

        Returns:
            The auxiliary loss.
        """
        if gate_logits is None or not isinstance(gate_logits, tuple):
            return 0

        if isinstance(gate_logits, tuple):
            compute_device = gate_logits[0].device
            concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
        with torch.no_grad():
            _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_mask = tx8_one_hot_opt(selected_experts, num_experts)
        if attention_mask is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
                expert_attention_mask, dim=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
                router_per_expert_attention_mask, dim=0
            )

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss * num_experts
    replace_tx8_one_hot_func_with_func(new_load_balancing_loss_func)

def replace_tx8_one_hot_func_with_func(new_func):
    import transformers.models.mixtral.modeling_mixtral as modeling_mixtral
    if torch_xla._XLAC.IsCurrentDeviceTx8() and hasattr(modeling_mixtral, 'load_balancing_loss_func'):
        print('!!!! replace torch.nn.functional.one_hot to', new_func.__name__)
        modeling_mixtral.load_balancing_loss_func = new_func

class Tx8MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, original_module):
        config = MixtralConfig()
        super().__init__()
        self.ffn_dim = original_module.ffn_dim
        self.hidden_dim = original_module.hidden_dim
        
        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False, dtype=torch.bfloat16)
        self.gate_proj.weight.data = copy.copy(original_module.w1.weight.data)
        self.down_proj.weight.data = copy.copy(original_module.w2.weight.data)
        self.up_proj.weight.data = copy.copy(original_module.w3.weight.data)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        current_hidden_states = self.down_proj(current_hidden_states)
        return current_hidden_states

def replace_MixtralBlockSparseTop2MLP_customcall(model):
    for name, module in model.named_children():
        if module.__class__.__name__ == "MixtralBlockSparseTop2MLP":
    #         # Assume the new module can be instantiated without arguments
    #         # print(*module.parameters())
            new_module = Tx8MixtralBlockSparseTop2MLP(module)
            setattr(model, name, new_module)
            del module
            print('!!!! replaceMixtralBlockSparseTop2MLP to Tx8MixtralBlockSparseTop2MLP')
        else:
            replace_MixtralBlockSparseTop2MLP_customcall(module)
    return model


""" 替换 MixtralSparseMoeBlock 类， tensor的多维索引引入的compare """
class Tx8MixtralSparseMoeBlock(nn.Module):
    def __init__(self, original_module:MixtralSparseMoeBlock):
        config = MixtralConfig(
                            hidden_size=original_module.hidden_dim,
                            intermediate_size=original_module.ffn_dim,
                            num_local_experts=original_module.num_experts,
                            num_experts_per_tok=original_module.top_k,
                            router_jitter_noise=original_module.jitter_noise,
                            )
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate_moe = nn.Linear(self.hidden_dim, self.num_experts, bias=False, dtype=torch.bfloat16)
        self.gate_moe.weight.data = copy.copy(original_module.gate.weight.data)
        self.experts = nn.ModuleList([Tx8MixtralBlockSparseTop2MLP(original_module.experts[ix]) for ix in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_moe(hidden_states)
        router_logits = router_logits.to(torch.bfloat16)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.bfloat16)
        with torch.no_grad():
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(torch.float32)
        selected_experts = selected_experts.to(torch.float32)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        dynamic_shape=self.dynamic_shape if hasattr(self,'dynamic_shape') else False
        expert_mask = tx8_one_hot_opt(selected_experts, num_classes=self.num_experts, dynamic_shape=dynamic_shape).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            my_idx = torch.tensor([expert_idx], dtype = torch.int32, device=hidden_states.device)
            my_mask = expert_mask.index_select(0, my_idx).squeeze(0)
            idx, top_x = torch.where(my_mask)
            idx = idx.to(torch.int32)
            top_x = top_x.to(torch.int32)
            idx = idx.to(torch.int32)
            top_x = top_x.to(torch.int32)
            current_state = torch.index_select(hidden_states, 0, top_x).unsqueeze(dim=0).reshape(-1, hidden_dim)
            routing_weights_flatten = routing_weights.flatten()
            n=routing_weights.size(1)
            indices=top_x*n+idx
            indices = indices.to(torch.int32)
            routing_weights=torch.index_select(routing_weights_flatten, 0 , indices).unsqueeze(-1)
            current_hidden_states = expert_layer(current_state) * routing_weights
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def replace_MixtralSparseMoeBlock_customcall(model):
    for name, module in model.named_children():
        if module.__class__.__name__ == "MixtralSparseMoeBlock":
    #         # Assume the new module can be instantiated without arguments
    #         # print(*module.parameters())
            new_module = Tx8MixtralSparseMoeBlock(module)
            setattr(model, name, new_module)
            del module
        else:
            replace_MixtralSparseMoeBlock_customcall(module)
    return model


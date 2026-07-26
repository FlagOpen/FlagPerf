import torch
import torch_xla
from tx8_util import IdentityToAllReduce,AllReduceToIdentity
from typing import List, Optional, Tuple, Union
from torch import nn



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

def LlamaRotaryEmbedding_forward(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    # Core RoPE block
    position_ids = position_ids.cpu()
    inv_freq = self.inv_freq.cpu()

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    # with torch.autocast(device_type=device_type, enabled=False):
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim = -1)
    cos = emb.cos()
    sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    # for batch size > 1, we slice cos and sin. e.g. [8, 2623, 128] => [2623, 128]
    if len(cos.shape) == 3:
        cos = cos[0]
        sin = sin[0]

    return cos.to(dtype=x.dtype).to(x.device), sin.to(dtype=x.dtype).to(x.device)

def replace_LlamaRotaryEmbedding_forward_with_new_func():
    import transformers.models.llama.modeling_llama as llama
    if  hasattr(llama, 'LlamaRotaryEmbedding'):
        layer = getattr(llama, 'LlamaRotaryEmbedding')
        layer.forward = LlamaRotaryEmbedding_forward
        print('!!!! replace transformers.models.modeling_llama.LlamaRotaryEmbedding.forward to LlamaRotaryEmbedding_forward SUCCESS')

def replace_tx8_rope():
    from torch_xla.tx8_custom_ops.rope import replace_tx8_rope_func
    replace_tx8_rope_func()
    print("replace rope SUCCESS")

def replace_tx8_mask():
    from torch_xla.tx8_custom_ops.causal_attention_mask import replace_tx8_causal_attention_mask_func
    replace_tx8_causal_attention_mask_func()
    print("replace mask SUCCESS")

def replace_tx8_rmsNorm(model):
    from torch_xla.tx8_custom_ops.replace_custom_ops import tx8_model_optimize
    model = tx8_model_optimize(model)
    print("replace rmsNorm SUCCESS")
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

#将_merge_input_ids_with_image_features函数移动到CPU上计算
def new_merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
    target_device = input_ids.device
    
    image_features = image_features.cpu()
    inputs_embeds = inputs_embeds.cpu()
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    labels = labels.cpu()
    
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
   
    special_image_token_mask = input_ids == self.config.image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)


    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype
    )
    final_attention_mask = torch.zeros(
        batch_size, max_embed_dim, dtype=attention_mask.dtype
    )
    if labels is not None:
        final_labels = torch.full(
            (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype
        )

    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    if labels is not None:
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]


    image_to_overwrite = torch.full(
        (batch_size, max_embed_dim), True, dtype=torch.bool
    )
    image_to_overwrite[batch_indices, text_to_overwrite] = False
    image_to_overwrite &= torch.logical_not(image_to_overwrite.cumsum(-1)) >= nb_image_pad[:, None]

    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError(
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
    indices_to_mask = new_token_positions[batch_indices, pad_indices]
    final_embedding[batch_indices, indices_to_mask] = 0

    if labels is None:
        final_labels = None

    return final_embedding.to(target_device), final_attention_mask.to(target_device), final_labels.to(target_device), position_ids.to(target_device)

def replace_merge_input_ids_with_image_features_with_new_func():
    import transformers.models.llava.modeling_llava as llava
    if hasattr(llava, 'LlavaForConditionalGeneration'):
        layer = getattr(llava, 'LlavaForConditionalGeneration')
        layer._merge_input_ids_with_image_features = new_merge_input_ids_with_image_features
        print('!!!! replace transformers.models.modeling_llava.LlavaForConditionalGeneration._merge_input_ids_with_image_features to new_merge_input_ids_with_image_features SUCCESS')

#替换transformers.models.llava.modeling_llava.py文件中LlavaForConditionalGeneration类的
#主要包括==、!=、>=、< 逻辑运算的替换以及部分索引的替换
import os
from transformers.training_args import logger
from transformers.utils import logging
logger = logging.get_logger(__name__) 
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

     
def equal(a,b):
    c =a-b
    d = torch.abs(c)
    e = torch.clamp(d,max=1)
    return 1-e


def not_equal(a,b):
    c = a-b
    d = torch.abs(c)
    e = torch.clamp(d,max=1,min=0)
    return e 


def greater_or_equal(a,b):
    c = a-b
    d = c+1
    e = torch.clamp(d,max=1,min=0)
    return e


def less_than(a,b):
    c = b-a
    d = c+1
    e = torch.clamp(d,max=1,min=0)
    return e

def new_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )
    legacy_processing = False
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        legacy_processing = (
            (input_ids.cpu() == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
            # less_than(equal(input_ids , self.config.image_token_index).sum(1).max() , self.config.image_seq_length)
        )or (input_ids.shape[-1] == 1 and pixel_values is not None)
        # )or (equal(input_ids.shape[-1] , 1) and pixel_values is not None)

    image_features = None
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
    if legacy_processing:
        logger.warning_once(
            "Expanding inputs for image tokens in LLaVa should be done in processing. "
            "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
            "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
            "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
        )
        
        if input_ids.shape[1] != 1:
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, labels
            )
            cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            
        else:
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
            batch_index, non_attended_tokens = torch.where(equal(first_layer_past_key_value.float().sum(-2) , 0))

            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            valid_indices = self.less_than(non_attended_tokens , extended_attention_mask.size(-1))
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]
    
    elif image_features is not None:
        n_image_tokens = (equal(input_ids , self.config.image_token_index)).sum(dim=-1)[0].item()
        n_image_features = image_features.shape[1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (self.equal(input_ids , self.config.image_token_index))
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        num_logits_to_keep=num_logits_to_keep,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        device = labels.device
        attention_mask = attention_mask.cpu()
        logits = logits.cpu()
        labels = labels.cpu()
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :]
            shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        max_padding_len = 1024
        shift_logits = nn.functional.pad(shift_logits, (0, 0, 0, max_padding_len - shift_logits.shape[0]), "constant", 0)
        shift_labels = nn.functional.pad(shift_labels, (0, max_padding_len - shift_labels.shape[0]), "constant", -100)

        shift_logits = shift_logits.to(device)
        shift_labels = shift_labels.to(device)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # loss=loss,
    # logits=logits,
    # past_key_values=outputs.past_key_values,
    # hidden_states=outputs.hidden_states,
    # attentions=outputs.attentions,
    # image_hidden_states=image_features if pixel_values is not None else None,
    # return loss,logits,past_key_values,hidden_states,attentions,image_hidden_states
    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )

def replace_forward_with_new_func():
    import transformers.models.llava.modeling_llava as llava
    if torch_xla._XLAC.IsCurrentDeviceTx8() and hasattr(llava, 'LlavaForConditionalGeneration'):
        layer = getattr(llava, 'LlavaForConditionalGeneration')
        layer.forward = new_forward
        print('!!!! replace transformers.models.modeling_llava.LlavaForConditionalGeneration.forward to new_forward SUCCESS')

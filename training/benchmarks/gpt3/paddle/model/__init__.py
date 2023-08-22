from memory_profiler import profile
from paddlenlp.transformers import GPTConfig, GPTForCausalLM

from .modeling_pp import GPTForCausalLMPipe


# @profile(precision=4, stream=open("memory_profiler_train_create.log", "w+"))
def create_model(config):
    gpt_config = GPTConfig(
        hidden_size=config.hidden_size,
        initializer_range=config.initializer_range,
        fuse_attention_qkv=config.fuse_attention_qkv,
        intermediate_size=config.intermediate_size,
        lm_shift_labels=config.lm_shift_labels,
        max_position_embeddings=config.max_position_embeddings,
        model_type=config.model_type,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        rms_norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        use_cache=config.use_cache,
        use_recompute=config.use_recompute,
        use_flash_attention=config.use_flash_attention,
        fp16_opt_level=config.fp16_opt_level,
    )

    model = GPTForCausalLM.from_pretrained(
        config.model_name_or_path,
        config=gpt_config,
        dtype="float16",
        load_state_as_np=True,
    )
    return gpt_config, model

from model.models.modeling import LlamaConfig, LlamaForCausalLM
from memory_profiler import profile
from contextlib import contextmanager
import paddle
# @profile(precision=4, stream=open("memory_profiler_train_create.log", "w+"))
def create_model(config):
    llama_config = LlamaConfig(hidden_size=config.hidden_size,
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
                               fp16_opt_level=config.fp16_opt_level)     
    @contextmanager
    def dtype_guard(dtype="float32"):
        origin_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype(dtype)
        try:
            yield
        finally:
            paddle.set_default_dtype(origin_dtype)

    if config.fp16:
        dtype = "float16"
    else:
        dtype = "float32"
       
    with dtype_guard(dtype):
        model = LlamaForCausalLM(llama_config)
    

    return llama_config, model
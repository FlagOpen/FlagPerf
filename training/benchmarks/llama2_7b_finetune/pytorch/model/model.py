from transformers import LlamaForCausalLM
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training
from typing import List
from dataclasses import dataclass, field
from dataclasses import asdict


@dataclass
class lora_config:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False


def generate_peft_config(train_config):
    configs = (lora_config, )
    peft_configs = (LoraConfig, )
    names = tuple(c.__name__.rstrip("_config") for c in configs)
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    config = configs[names.index(train_config.peft_method)]()
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    return peft_config


def get_llama_model(train_config):
    use_cache = False if train_config.enable_fsdp else None
    model_path = train_config.data_dir + "/" + train_config.model_name
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=None,
        device_map=None,
        use_cache=use_cache,
    )

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if train_config.use_fp16:
        model.half()
    model.to("cuda")
    return model

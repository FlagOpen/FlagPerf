from .configuration_baichuan import BaichuanConfig
from .modeling_baichuan import BaichuanForCausalLM

def get_baichuan_model(model_config_dir, flashattn):

    config = BaichuanConfig.from_pretrained(model_config_dir)
    config._flash_attn_2_enabled = flashattn
    model = BaichuanForCausalLM(config)
    # model.enable_checkpointing()
    # 激活值重计算 checkpointing ckpting ZeRO-R
# proc:Gemory xvqiu down 
# cons: jisuanliang + 34%

    return model


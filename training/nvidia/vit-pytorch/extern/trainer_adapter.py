import config
from torch import nn
from driver.dist_pytorch import main_proc_print

def convert_model(model: nn.Module) -> nn.Module:
    return model

def model_to_fp16(model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model

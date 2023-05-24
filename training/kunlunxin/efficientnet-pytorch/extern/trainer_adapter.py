from torch import nn
from driver.dist_pytorch import main_proc_print


def convert_model(args, model: nn.Module) -> nn.Module:
    return model

def model_to_fp16(args, model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model
def create_grad_scaler(args):
    return None

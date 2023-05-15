import config
from torch import nn
import torch.distributed as dist
from driver.dist_pytorch import main_proc_print
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex.parallel import DistributedDataParallel as ApexDDP
    has_apex = True
except ImportError:
    has_apex = False


def convert_model(model: nn.Module) -> nn.Module:
    return model

def model_to_fp16(model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model

def model_to_ddp(model: nn.Module, use_amp) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            main_proc_print("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            main_proc_print("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model)
        # NOTE: EMA model does not need to be wrapped by DDP
    return model

from torch import nn
import torch.distributed as dist

def model_to_fp16(model: nn.Module):
    return model

def model_to_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    return model

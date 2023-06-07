import torch.distributed as dist
import torch

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
      
from common.fairseq.optim.fused_adam import get_fused_adam_class
from driver.dist_pytorch import main_proc_print


def convert_model(model: nn.Module) -> nn.Module:
    return model


def model_to_fp16(model):
    return model

def create_optimizer(model, args):

    kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'adam' and not (args.fp16 or args.bf16):
        kw.update({'betas': args.adam_betas, 'eps': args.adam_eps})
        optimizer = torch.optim.Adam(model.parameters(), **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    return optimizer

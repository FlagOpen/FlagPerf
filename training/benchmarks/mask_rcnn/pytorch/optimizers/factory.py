import torch
from torch import nn
from torch.optim import Optimizer


def create_optimizer(model: nn.Module, args) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

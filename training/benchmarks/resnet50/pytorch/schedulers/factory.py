from torch.optim.lr_scheduler import StepLR
from typing import Any


def create_scheduler(optimizer: Any, args) -> Any:
    scheduler = StepLR(optimizer,
                       step_size=args.lr_step_size,
                       gamma=args.lr_gamma)
    return scheduler

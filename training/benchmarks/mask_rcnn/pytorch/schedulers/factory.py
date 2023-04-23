from typing import Any
from torch.optim.lr_scheduler import MultiStepLR


def create_scheduler(optimizer, args) -> Any:
    """create_scheduler"""
    scheduler = MultiStepLR(optimizer,
                            milestones=args.lr_steps,
                            gamma=args.lr_gamma)
    return scheduler
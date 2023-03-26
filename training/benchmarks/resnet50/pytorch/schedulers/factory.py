import config

from torch.optim.lr_scheduler import StepLR
from typing import Any


def create_scheduler(optimizer: Any) -> Any:
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    return scheduler

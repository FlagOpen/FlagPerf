import torch
from typing import Any


def create_optimizer(model, config) -> Any:
    optimizer = torch.optim.SGD(model.parameters(), config.learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay_rate)
    return optimizer

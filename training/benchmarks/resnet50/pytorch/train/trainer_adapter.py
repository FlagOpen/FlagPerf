import os
import sys
import torch

from torch.optim import Optimizer
from torch import nn, Tensor
from typing import Any, Tuple


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))


def convert_model(model: nn.Module) -> nn.Module:
    return model


def create_optimizer(model, config) -> Any:
    optimizer = torch.optim.SGD(model.parameters(), config.learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay_rate)
    return optimizer


def model_to_fp16(model: nn.Module, optimizer: Optimizer) -> Tuple[nn.Module, Optimizer]:
    return model, optimizer


def model_to_ddp(model: nn.Module) -> nn.Module:
    return model


def create_grad_scaler():
    return None


def backward(step: int, loss: Tensor, optimizer, **kwarg):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return

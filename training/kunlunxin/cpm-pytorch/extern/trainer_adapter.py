from model.models import gpt2_get_params_for_weight_decay_optimization

from torch import nn
from torch.optim import Optimizer
from typing import Tuple
from driver.dist_pytorch import main_proc_print


def convert_model(config, model: nn.Module) -> nn.Module:
    return model


def create_optimizer(config, model):
    param_groups = gpt2_get_params_for_weight_decay_optimization(model)
    from torch.optim import Adam
    optimizer = Adam(param_groups,
                     lr=config.learning_rate,
                     weight_decay=config.weight_decay_rate)

    return optimizer


def model_to_fp16(config, model: nn.Module,
                  optimizer: Optimizer) -> Tuple[nn.Module, Optimizer]:
    return model, optimizer

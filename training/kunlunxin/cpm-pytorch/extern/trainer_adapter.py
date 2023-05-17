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

class Wrap_Module(nn.Module):

    def __init__(self, module):
        super(Wrap_Module, self).__init__()
        self.add_module('module', module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

def model_to_fp16(config, model: nn.Module,
                  optimizer: Optimizer) -> Tuple[nn.Module, Optimizer]:
    # we don't support fp16 now, but we should follow the FP16_Module behavior
    model = Wrap_Module(model)
    return model, optimizer
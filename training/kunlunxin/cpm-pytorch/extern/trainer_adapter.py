from model.models import gpt2_get_params_for_weight_decay_optimization

from torch import nn
from torch.optim import Optimizer
from typing import Tuple

from driver.dist_pytorch import main_proc_print
from model.fp16 import FP16_Module
from model.fp16 import FP16_Optimizer

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
    args = config
    if args.fp16:
        model = FP16_Module(model)
        optimizer = FP16_Optimizer(optimizer,
                                static_loss_scale=args.loss_scale,
                                dynamic_loss_scale=args.dynamic_loss_scale,
                                dynamic_loss_args={
                                    'scale_window': args.loss_scale_window,
                                    'min_scale': args.min_scale,
                                    'delayed_shift': args.hysteresis
                                })
        for layer in model.modules():
            if isinstance(layer, nn.modules.normalization.LayerNorm):
                layer = layer.float()
    return model, optimizer

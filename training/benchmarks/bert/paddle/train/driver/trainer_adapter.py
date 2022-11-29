import config, optimizers

import paddle
import paddle.nn as nn

from typing import Tuple


def convert_model(model: nn.Layer) -> nn.Layer:
    return model


def create_optimizer(model: nn.Layer, lr_scheduler):
    named_params = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm', 'norm']

    decay_params = [
        p.name for n, p in named_params if not any(nd in n for nd in no_decay)
    ]

    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in named_params if not any(nd in n for nd in no_decay)],
        'weight_decay':
        config.weight_decay_rate
    }, {
        'params':
        [p for n, p in named_params if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    optimizer = optimizers.create_optimizer(
        name='adamw',
        params=optimizer_grouped_parameters,
        config=config,
        decay_params=decay_params,
        lr_scheduler=lr_scheduler)
    return optimizer


def model_to_fp16(model: nn.Layer, optimizer):
    return model, optimizer


def model_to_ddp(model: nn.Layer) -> nn.Layer:
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    return model


def create_grad_scaler():
    return None


def backward(step: int, loss, optimizer, **kwarg):
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    return

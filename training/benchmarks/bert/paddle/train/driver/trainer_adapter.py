# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    need_update = step % config.gradient_accumulation_steps == 0
    if need_update:
        optimizer.step()
        optimizer.clear_grad()
    return

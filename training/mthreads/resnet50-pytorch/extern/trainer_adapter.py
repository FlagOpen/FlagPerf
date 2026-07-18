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

import torch
import torch_musa
import config
from driver import dist_pytorch


def convert_model(model):
    model.to(memory_format=torch.channels_last)
    return model


def create_grad_scaler():
    """create_grad_scaler for mixed precision training"""
    scaler = torch_musa.amp.GradScaler() if config.amp else None
    return scaler


def train_step(model, batch, optimizer, scaler=None):
    """train one step"""
    images, target = batch
    criterion = torch.nn.CrossEntropyLoss()
    if scaler:
        with torch.musa.amp.autocast(enabled=True):
            output = model(images)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return loss

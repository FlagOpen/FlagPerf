# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
from torch.optim.lr_scheduler import _LRScheduler


class MixedAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, lr, warmup_step, max_step, eta_min):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_step = warmup_step
        self.max_step = max_step
        self.eta_min = eta_min
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step, eta_min)
        self.train_step = 0

    def get_lr(self):
        if self.train_step >= self.warmup_step:
            return self.scheduler.get_lr()

        curr_lr = self.lr * self.train_step / self.warmup_step
        return curr_lr

    def step(self, step=None):
        if step is None:
            self.train_step += 1
        else:
            self.train_step = step

        for g in self.optimizer.param_groups:
            lr = self.get_lr()
            lr = lr[0] if isinstance(lr, list) else lr
            g["lr"] = lr

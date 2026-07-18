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
from .base import LRScheduler


class LinearWarmupPolyDecayScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self,
                 optimizer,
                 start_warmup_steps,
                 warmup_steps,
                 total_steps,
                 end_learning_rate=0.0,
                 degree=1.0,
                 last_epoch=-1):
        self.num_warmup_updates = warmup_steps
        self.start_warmup_steps = start_warmup_steps
        self.total_steps = total_steps
        self.end_learning_rate = end_learning_rate
        self.degree = degree
        super(LinearWarmupPolyDecayScheduler,
              self).__init__(optimizer, last_epoch)

        if self.last_epoch <= 0:
            self.last_epoch = 0

    def step(self, epoch=None):
        param_group = self.optimizer.param_groups[0]
        self.last_epoch = self.optimizer._step_count + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        mod_step = self.last_epoch - self.start_warmup_steps
        if mod_step < self.num_warmup_updates:
            progress = mod_step / self.num_warmup_updates
            return [(base_lr * progress) for base_lr in self.base_lrs]
        else:
            progress = min(self.last_epoch / self.total_steps, 1.0)
            return [(base_lr - self.end_learning_rate) *
                    (1 - progress)**self.degree + self.end_learning_rate
                    for base_lr in self.base_lrs]

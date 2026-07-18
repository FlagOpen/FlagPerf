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

import sys
from paddle.optimizer.lr import LRScheduler


class LinearWarmupPolyDecayScheduler(LRScheduler):

    def __init__(self,
                 startup_warmup_steps,
                 warmup_steps,
                 total_steps,
                 base_lr,
                 end_lr=0.0,
                 degree=1.0,
                 last_epoch=-1):
        self.startup_warmup_steps = startup_warmup_steps
        self.offset_step = int(startup_warmup_steps == 0)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.degree = degree
        super(LinearWarmupPolyDecayScheduler,
              self).__init__(learning_rate=base_lr, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        mod_step = step - self.offset_step - self.startup_warmup_steps
        if mod_step < self.warmup_steps:
            p = mod_step / (self.warmup_steps + 1e-6)
            lr = self.base_lr * p
        else:
            p = min(1, (step - self.offset_step) / self.total_steps)
            lr = (self.base_lr - self.end_lr) * (1 -
                                                 p)**self.degree + self.end_lr
        return lr

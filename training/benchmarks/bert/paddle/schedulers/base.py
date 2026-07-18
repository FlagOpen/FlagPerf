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
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1):
        # Check if using mixed precision training
        self.mixed_training = False
        base_optimizer = optimizer

        # Check that optimizer param is valid
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        super(LRScheduler, self).__init__(base_optimizer, last_epoch)

    def step(self, epoch=None):
        # Set the current training step
        # ('epoch' is used to be consistent with _LRScheduler)
        if self.mixed_training:
            # The assumption is that the step will be constant
            state_dict = self.optimizer.state[self.optimizer.param_groups[0]
                                              ['params'][0]]
            if 'step' in state_dict:
                self.last_epoch = state_dict['step'] + 1
            else:
                self.last_epoch = 1
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import math

from common.fairseq.optim.adam import FairseqAdam
from common.fairseq.optim.fp16_optimizer import FP16Optimizer
from common.fairseq.optim.fused_adam import get_fused_adam_class
from common.utils import print_once


def lr_poly_policy(step, optimizer, lr, initial_lr_scale=0.0,
                   final_lr_scale=0.0, warmup_steps=1000, hold_steps=0,
                   num_steps=None, power=1.0):
    """Polynomial decay LR policy with an optional hold period."""
    assert step >= 1
    assert num_steps is not None
    assert power is not None

    start_lr = initial_lr_scale * lr
    end_lr = final_lr_scale * lr

    if step <= warmup_steps:
        new_lr = start_lr + (step) / warmup_steps * (lr - start_lr)
    elif step <= warmup_steps + hold_steps:
        new_lr = lr
    elif warmup_steps + hold_steps < step <= num_steps:
        remain = 1 - (step - warmup_steps) / (num_steps - warmup_steps)
        new_lr = (lr - end_lr) * remain ** power + end_lr
    else:
        new_lr = end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


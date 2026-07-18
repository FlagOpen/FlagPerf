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

from schedulers.optimizer_param_scheduler import OptimizerParamScheduler
import config

def create_scheduler(optimizer):
    """Build the learning rate scheduler."""

    # Iteration-based training.
    if config.max_steps:
        config.lr_decay_iters = config.max_steps
        lr_decay_steps = config.lr_decay_iters * config.global_batch_size
        wd_incr_steps = config.max_steps* config.global_batch_size
        if config.lr_warmup_fraction is not None:
            lr_warmup_steps = config.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = config.lr_warmup_iters * config.global_batch_size
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.start_weight_decay,
        end_wd=config.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=config.weight_decay_incr_style,
        )

    return opt_param_scheduler

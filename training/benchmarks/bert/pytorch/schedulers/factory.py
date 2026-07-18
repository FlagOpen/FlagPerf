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

import config

from .linear_warmup_poly_scheduler import LinearWarmupPolyDecayScheduler
from .linear_warmup_scheduler import LinearWarmUpScheduler


def create_scheduler(optimizer, scheduler="poly"):
    if config.warmup_proportion == 0:
        warmup_steps = config.warmup_steps
        warmup_start = config.start_warmup_step
    else:
        warmup_steps = int(config.max_steps * config.warmup_proportion)
        warmup_start = 0

    if scheduler == "linear":
        return LinearWarmUpScheduler(optimizer, warmup_steps, config.max_steps)

    if scheduler == "poly":
        return LinearWarmupPolyDecayScheduler(optimizer,
                                              start_warmup_steps=warmup_start,
                                              warmup_steps=warmup_steps,
                                              total_steps=config.max_steps,
                                              end_learning_rate=0.0,
                                              degree=1.0)

    raise ValueError(f"Not found scheduler {scheduler}.")

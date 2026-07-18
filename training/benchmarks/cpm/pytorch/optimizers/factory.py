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

from torch.optim import AdamW


def create_optimizer(name: str, params, config):
    name = name.lower()

    if name == "adamw":
        return AdamW(params,
                     lr=config.learning_rate,
                     betas=(config.beta_1, config.beta_2),
                     weight_decay=config.weight_decay_rate)

    raise RuntimeError(f"Not found optimizer {name}.")

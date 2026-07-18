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

import paddle


def create_optimizer(name: str, params, config, decay_params, lr_scheduler):
    name = name.lower()
    if name == "lamb":
        return paddle.optimizer.Lamb(
            parameters=params,
            learning_rate=lr_scheduler,
            beta1=config.opt_lamb_beta_1,
            beta2=config.opt_lamb_beta_2,
            epsilon=1e-6,
            #lamb_weight_decay=config.weight_decay_rate,
            # #exclude_from_weight_decay_fn=lambda x: x in decay_params,
        )
        raise Exception("Not Implementation Lamb Error")

    if name == "adamw":
        return paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=params,
            #weight_decay=config.weight_decay_rate,
            beta1=config.opt_lamb_beta_1,
            beta2=config.opt_lamb_beta_2,
            epsilon=1e-6
            #apply_decay_param_fun=lambda x: x in decay_params,
        )

    raise RuntimeError(f"Not found optimier {name}.")

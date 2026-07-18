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

from .models.model_helper import get_model_config, get_model


def create_model(args):
    model_config = get_model_config(args)
    uniform_initialize_bn_weight = not args.disable_uniform_initialize_bn_weight
    model = get_model(
        model_config,
        cpu_run=False,
        uniform_initialize_bn_weight=uniform_initialize_bn_weight)
    return model


def create_model_config(args):
    model_config = get_model_config(args)
    return model_config

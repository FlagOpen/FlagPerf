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

from .layers import LayerNorm


def convert_model(config, model):
    state_dict = model.state_dict()
    state_dict = remap_attn_parameters(state_dict)
    for i in range(config.num_layers):
        model.transformer.layers[i].input_layernorm = LayerNorm(
            config.hidden_size, config.layernorm_epsilon)
        model.transformer.layers[i].post_attention_layernorm = LayerNorm(
            config.hidden_size, config.layernorm_epsilon)
    model.transformer.final_layernorm = LayerNorm(config.hidden_size,
                                                  config.layernorm_epsilon)

    model.load_state_dict(state_dict, strict=True)
    return model


def remap_attn_parameters(model_dict):
    return model_dict

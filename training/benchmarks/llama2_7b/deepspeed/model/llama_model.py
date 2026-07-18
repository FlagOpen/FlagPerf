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

from transformers import LlamaForCausalLM, LlamaConfig


def get_llama_model(model_config_dir, flashattn):

    config = LlamaConfig.from_pretrained(model_config_dir)
    config._flash_attn_2_enabled = flashattn
    model = LlamaForCausalLM(config)

    return model

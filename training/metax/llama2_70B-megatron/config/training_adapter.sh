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

export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM-FlagScale
CODE_PATH="/workspace/Megatron-LM-FlagScale/pretrain_llama.py"
RECOMPUTE_ARGS="
    --pipline-num-layers-list 9 9 10 10 10 11 11 10
"

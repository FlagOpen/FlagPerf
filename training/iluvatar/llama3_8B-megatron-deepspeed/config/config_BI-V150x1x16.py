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

tokenizer_path = "/data1/user_homes/lisen/codes/FlagPerf/training/benchmarks/llama3_8B/megatron-deepspeed/tokenizer_llama3.model"
localbs = 1
train_steps = 300
theoryflops = 192000000000000.0
megatron_path = "/workspace/megatron-deepspeed" # need to be aligned with DockerFile. In iluvatar, it's /workspace/ + Megatron-LM
tensor_parallel = 1
pipeline_parallel = 8

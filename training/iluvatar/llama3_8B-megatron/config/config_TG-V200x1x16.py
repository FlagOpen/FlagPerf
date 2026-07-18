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

tokenizer_path = "/home/zhiyuan/test/llama3-8b/tokenizer.model"
localbs = 1  #micro-batch-size
train_steps = 300  ##训练迭代次数
theoryflops = 340000000000000.0
megatron_path = "/usr/local/lib/python3.10/dist-packages/megatron" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 1  
pipeline_parallel = 4 

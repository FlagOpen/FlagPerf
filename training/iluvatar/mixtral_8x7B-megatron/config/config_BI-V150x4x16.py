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

mixtral_iluvatar_path = "/data1/user_homes/gengyang/code/FlagPerf/data_dir"
tokenizer_path = mixtral_iluvatar_path + "/flagscale-iluvatar-mixtral/data_dir/Qwen1___5-7B-Chat-GPTQ-Int8"
localbs = 1  #micro-batch-size
train_steps = 100  ##训练迭代次数
theoryflops = 192000000000000.0
megatron_path = mixtral_iluvatar_path + "/flagscale-iluvatar-mixtral"#"/workspace/Megatron-LM" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 4  #四机为4,非四机暂设为2
pipeline_parallel = 2 

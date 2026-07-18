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

target_mlm_accuracy = 0.67
gradient_accumulation_steps = 1
max_steps = 15000
start_warmup_step = 0
warmup_proportion = 0
warmup_steps = 0

learning_rate = 1e-4
weight_decay_rate = 0.01
opt_lamb_beta_1 = 0.9
opt_lamb_beta_2 = 0.999
train_batch_size = 12
eval_batch_size = train_batch_size
max_samples_termination = 4500000
cache_eval_data = False

seed = 9031

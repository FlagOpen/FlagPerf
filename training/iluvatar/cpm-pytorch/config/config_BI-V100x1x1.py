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

from config_common import *

fp16 = True
dist_backend = "nccl"
target_embedding_average = 0.8

gradient_accumulation_steps = 1

train_batch_size = 32
eval_batch_size = train_batch_size
max_steps = 60000
max_samples_termination = 439126000

warmup = 0.2
learning_rate = 0.0005

beta_1: float = 0.9
beta_2: float = 0.99
eps: float = 1e-08

seed = 23333
training_event = None

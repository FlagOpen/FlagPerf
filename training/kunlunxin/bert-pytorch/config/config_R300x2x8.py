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

fp16 = False
dist_backend = "xccl"
gradient_accumulation_steps = 4
train_batch_size = 8
max_steps = 30000
start_warmup_step = 0
warmup_proportion = 0
warmup_steps = 0

distributed_lamb = False
learning_rate = 0.00035
weight_decay_rate = 0.01
opt_lamb_beta_1 = 0.9
opt_lamb_beta_2 = 0.999

eval_batch_size = train_batch_size
max_samples_termination = 45000000
cache_eval_data = False

fused_gelu_bias = False
fused_mha = False
dense_seq_output = False
exchange_padding = False

dwu_num_rs_pg = 1
dwu_num_ar_pg = 1
dwu_num_blocks = 1

seed = 9031

from torch_xmlir.amp import GradScaler
grad_scaler = GradScaler(enabled=False)

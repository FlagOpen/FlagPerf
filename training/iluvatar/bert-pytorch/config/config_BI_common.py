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
from torch.cuda.amp import GradScaler
import os

grad_scaler = GradScaler(init_scale=float(os.getenv("INIT_LOSS_SCALE", 2**20)),
                         growth_interval=2000, enabled=True)

fp16 = True
ddp_type = "apex"
dist_backend = "nccl"

train_batch_size = 20

fused_gelu_bias = True
fused_mha = True
unpad = True
unpad_fmha = False
dense_seq_output = True
exchange_padding = True

dwu_num_rs_pg = 1
dwu_num_ar_pg = 1
dwu_num_blocks = 1

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

# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "mobilenetv2"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "val"

# =========================================================
# loss scale
# =========================================================
lr: float = 0.045
weight_decay: float = 0.00004
momentum: float = 0.9
lr_steps: list = 1
lr_gamma: float = 0.98

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 8
eval_batch_size: int = 8

# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L193
target_acc1: float = 68.6
# https://github.com/pytorch/vision/tree/main/references/classification
max_epoch: int = 300


do_train = True
fp16 = False
amp: bool = False
distributed: bool = True

# =========================================================
# utils
# =========================================================
seed: int = 41
dist_backend: str = 'nccl'
num_workers: int = 16
device: str = None

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 100
print_freq: int = 100
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1

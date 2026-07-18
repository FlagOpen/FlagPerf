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
name: str = "faster_rcnn"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters

# =========================================================
# data
# =========================================================
train_data: str = "train"
eval_data: str = "val"

# use torch/hub/checkpoints/resnet50-0676ba61 as backbone weights
# no init weights for other parts of faster_rcnn

# =========================================================
# Model
# =========================================================
aspect_ratio_group_factor: int = 3
num_classes: int = 90

# =========================================================
# loss scale
# =========================================================
lr: float = 0.02
weight_decay: float = 1e-4
momentum: float = 0.9
lr_steps: list = [16, 22]
lr_gamma: float = 0.1

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 2
eval_batch_size: int = 2

target_mAP: float = 0.36
max_epoch: int = 26

do_train = True
fp16 = False
distributed: bool = True
warmup = 0.1

# =========================================================
# utils
# =========================================================
seed: int = 0
dist_backend: str = 'nccl'
num_workers: int = 4
device: str = None

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 1000
print_freq: int = 1000
n_device: int = 1
amp: bool = False
sync_bn: bool = False
gradient_accumulation_steps: int = 1

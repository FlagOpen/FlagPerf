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

vendor = "nvidia"
dist_backend = "nccl"

lr = 0.5
lr_scheduler = "cosineannealinglr"
lr_warmup_epochs = 5
lr_warmup_method = "linear"
auto_augment = "ta_wide"
random_erase = 0.1
label_smoothing = 0.1
mixup_alpha = 0.2
cutmix_alpha = 1.0
weight_decay = 0.00002
norm_weight_decay = 0.0
ra_sampler = True
ra_reps = 4
epochs = 600
num_workers = 8

# efficientnet_v2_s
TRAIN_SIZE = 300
train_crop_size = TRAIN_SIZE
EVAL_SIZE = 384
val_crop_size = EVAL_SIZE
val_resize_size = EVAL_SIZE

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

vendor = "metax"
dist_backend = "nccl"

epochs = 300
opt = "adamw"
lr = 0.003
weight_decay = 0.3
lr_scheduler = "cosineannealinglr"
lr_warmup_method = "linear" 
lr_warmup_epochs = 30
lr_warmup_decay = 0.033 
amp = False
label_smoothing = 0.11
mixup_alpha = 0.2
auto_augment = "ra"
clip_grad_norm = 1
ra_sampler = True
cutmix_alpha = 1.0

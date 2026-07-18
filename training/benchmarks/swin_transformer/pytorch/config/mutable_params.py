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

"""mutable_params defines parameters that can be replaced by vendor"""
mutable_params = [
    "vendor",
    "model_type",
    "data_dir",
    "dist_backend",
    "gradient_accumulation_steps",
    "train_batch_size",
    "train_warmup_epochs",
    "seed",
    "train_optimizer_name",
    "train_lr_scheduler_name",
    "train_lr_scheduler_decay_rate",
    "data_num_workers",
    "device",
    "local_rank",
    "do_train",
    "log_freq",
]

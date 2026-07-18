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

# Common arguments
theoryflops = 312000000000000.0

# pretrain arguments
pretrain_per_device_train_batch_size = 32
pretrain_gradient_accumulation_steps = 1


# finetune arguments
finetune_per_device_train_batch_size = 16
finetune_gradient_accumulation_steps = 1
output_dir_finetune = "Output/checkpoints_finetune/llava-v1.5-7b"

# eval arguments
mmmu_data_path = "MMMU/MMMU"

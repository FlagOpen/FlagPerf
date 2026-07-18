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

mutable_params = [
    "split",
    "max_seq_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "use_flash_attention",
    "use_fused_rms_norm",
    "fp16",
    "fp16_opt_level",
    "gradient_accumulation_steps",
    "max_steps",
    "eval_steps",
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "warmup_steps",
    "seed",
    "sharding",
    "recompute",
]

mutable_params += ["local_rank", "do_train", "input_dir", "logging_steps"]

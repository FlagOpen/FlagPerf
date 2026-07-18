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
    "model_name_or_path",
    "tokenizer_name_or_path",
    "input_dir",
    "output_dir",
    "split",
    "max_seq_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "use_flash_attention",
    "use_fused_rms_norm",
    "fp16",
    "fp16_opt_level",
    "scale_loss",
    "learning_rate",
    "min_learning_rate",
    "max_steps",
    "save_steps",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "logging_steps",
    "dataloader_num_workers",
    "eval_steps",
    "disable_tqdm",
    "continue_training",
    "recompute",
    "do_train",
    "do_eval",
    "data_impl",
    "gradient_accumulation_steps",
    "tensor_parallel_degree",
    "pipeline_parallel_degree",
    "virtual_pp_degree",
    "sequence_parallel",
    "distributed_dataloader",
]

mutable_params += ["local_rank", "dist_backend"]

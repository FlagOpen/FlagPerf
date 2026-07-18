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
    "train_batch_size", "eval_batch_size", "learning_rate",
    "weight_decay_rate", "opt_lamb_beta_1", "opt_lamb_beta_2", "max_steps",
    "max_samples_termination", "warmup_proportion", "warmup_steps",
    "start_warmup_step", "dist_backend", "seed", "gradient_accumulation_steps",
    "fp16", "loss_scale", "exchange_padding", "enable_fuse_dropout",
    "disable_fuse_mask", "fused_gelu_bias", "fused_dropout_add",
    "dense_seq_output"
    #"cache_eval_data"
]

mutable_params += [
    "use_cuda", "local_rank", "do_train", "data_dir", "log_freq"
]

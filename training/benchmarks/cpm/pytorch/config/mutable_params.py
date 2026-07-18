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
    "fp16", "dist_backend", "gradient_accumulation_steps", "train_batch_size",
    "eval_batch_size", "max_steps", "max_samples_termination", "warmup",
    "warmup_steps", "lr_decay_iters", "learning_rate", "weight_decay_rate",
    "beta_1", "beta_2", "eps", "seed", "target_embedding_average",
    "attention_dropout", "hidden_dropout", "loss_scale", "dynamic_loss_scale",
    "hysteresis", "loss_scale_window", "min_scale",
    "use_gradient_as_bucket_view", "num_workers", "vendor"
]

mutable_params += ["local_rank", "do_train", "data_dir", "log_freq"]

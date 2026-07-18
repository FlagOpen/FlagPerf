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
    "amp",
    "dist_backend",
    "train_batch_size",
    "eval_batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
    "loss_scale",
    "loss_scale_window",
    "min_scale",
    "num_workers",
    "distributed",
    "init_checkpoint",
    "vendor",
    'cudnn_benchmark',
    'cudnn_deterministic'
]

mutable_params += ["local_rank", "do_train", "data_dir"]

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
    'train_data', 'eval_data', 'init_checkpoint', 'train_batch_size',
    'eval_batch_size', 'dist_backend', 'vendor', 'local_rank', 'do_train',
    'data_dir', 'log_freq', 'output_dir', 'resume', 'gradient_accumulation_steps',
    'cudnn_benchmark',
    'cudnn_deterministic' 
]
mutable_params += [
    'epochs', 'opt', 'lr', 'weight_decay', 'lr_scheduler', 'lr_warmup_method',
    'lr_warmup_epochs', 'lr_warmup_decay', 'amp', 'label_smoothing',
    'mixup_alpha', 'auto_augment', 'clip_grad_norm', 'ra_sampler', 
    'cutmix_alpha'
]

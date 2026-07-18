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
    'data_dir', 'vendor', 'max_update', 'max_epoch', 'local_rank', 'dist_backend', 'distributed_world_size',
    'distributed_rank', 'lr', 'seed'
]
mutable_params += [
    'source_lang', 'target_lang', 'raw_text', 'left_pad_source', 'left_pad_target',
    'max_source_positions', 'max_target_positions', 'save_dir', 'restore_file', 'max_tokens', 'max_sentences'
]

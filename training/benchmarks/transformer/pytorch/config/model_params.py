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


decoder_attention_heads = 16
decoder_embed_dim = 1024
decoder_embed_path = None
decoder_ffn_embed_dim = 4096
decoder_layers = 6
decoder_learned_pos = False
decoder_normalize_before = True

encoder_attention_heads = 16
encoder_embed_dim = 1024
encoder_embed_path = None
encoder_ffn_embed_dim = 4096
encoder_layers = 6
encoder_learned_pos = False
encoder_normalize_before = True

fuse_dropout_add = False
fuse_layer_norm = False
fuse_relu_dropout = False

max_len_a = 0
max_len_b = 200
max_positions = (1024, 1024)
max_source_positions = 1024
max_target_positions = 1024
min_len = 1

dropout = 0.1

label_smoothing = 0.1
left_pad_source = True
left_pad_target = False
pad_sequence = 1

attention_dropout = 0.1
beam = 4
bpe_codes = None
buffer_size = 64
clip_norm = 0.0

raw_text = False
relu_dropout = 0.1
remove_bpe = "@@ "
replace_unk = None

nbest = 1
no_beamable_mm = False
no_early_stop = False
no_epoch_checkpoints = False
no_save = False
no_token_positional_embeddings = False

print_alignment = False
prefix_size = 0

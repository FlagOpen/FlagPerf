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

from driver import dist_pytorch
from .layers.transformer import GLMTransformer


def convert_model(model, config):
    if dist_pytorch.get_rank() == 0:
        print("convert model", flush=True)
    model = model.cpu()
    state_dict = model.state_dict()
    transformer_layer = GLMTransformer(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        max_sequence_length=config.max_seq_length,
        max_memory_length=config.max_memory_length,
        embedding_dropout_prob=config.hidden_dropout,
        attention_dropout_prob=config.attention_dropout,
        output_dropout_prob=config.hidden_dropout,
        checkpoint_activations=config.checkpoint_activations)
    model.model.transformer = transformer_layer
    model.load_state_dict(state_dict, strict=True)
    model = model.to(config.device)
    return model

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

import torch

from .models.modeling import GLMModel, GLMForMultiTokenCloze, FP16_Module


def create_model(config):
    model = GLMModel(num_layers=config.num_layers,
                     vocab_size=config.vocab_size,
                     hidden_size=config.hidden_size,
                     num_attention_heads=config.num_attention_heads,
                     embedding_dropout_prob=config.hidden_dropout,
                     attention_dropout_prob=config.attention_dropout,
                     output_dropout_prob=config.hidden_dropout,
                     max_sequence_length=config.max_position_embeddings,
                     max_memory_length=config.max_memory_length,
                     checkpoint_activations=config.checkpoint_activations)

    model = GLMForMultiTokenCloze(model)
    return model

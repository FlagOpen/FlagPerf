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

from model.models.modeling import BertForPretraining
from model.models.modeling import BertConfig, BertForPreTraining


def create_model(config):
    config.resume_step = 0

    bert_config = BertConfig.from_json_file(config.bert_config_path)
    bert_config.fused_gelu_bias = config.fused_gelu_bias
    bert_config.dense_seq_output = config.dense_seq_output
    bert_config.fuse_dropout = config.enable_fuse_dropout
    bert_config.fused_dropout_add = config.fused_dropout_add

    # Padding for divisibility by 8
    if bert_config.vocab_size % 8 != 0:
        bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)

    model = BertForPreTraining(bert_config)
    return bert_config, model

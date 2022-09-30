# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT-2 model."""

import torch
import torch.nn.functional as F
from model import models

def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class GPT2Model(torch.nn.Module):
    """GPT-2 Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True):

        super(GPT2Model, self).__init__()

        init_method = init_method_normal(std=0.02)

        # Shape of word_embeddings.weight: [vocab_size, hidden_size]
        self.word_embeddings =torch.nn.Embedding(vocab_size, hidden_size)
        init_method(self.word_embeddings.weight)

        # Position embedding (serial).
        # Shape of position_embeddings.weight:[max_sequence_length, hidden_size]. And max_sequence_length is max length of sentence that model supported.
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                      hidden_size)
        # Initialize the position embeddings.
        init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = models.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers)

    def forward(self, input_ids, position_ids, attention_mask):

        # Embeddings.
        #input_ids的形状[b,s]
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #embeddings的形状[b,s,hidden_size]
        embeddings = words_embeddings + position_embeddings

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        # Transformer.
        #transformer_output的形状[b,s,hidden_size]
        transformer_output = self.transformer(embeddings, attention_mask)

        #logits_parallel 的形状[b,s,vocab_size_per_part]
        logits = F.linear(transformer_output,
                                   self.word_embeddings.weight)

        #return 的形状[b,s,vocab_size]
        return logits


def judge_name(name):
    if "embedding" in name:
        return False
    for i in range(28):
        if str(i) in name:
            return False
    return True

def gpt2_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for name, module_ in module.named_modules():
        # if not judge_name(name):
        #     continue
        if isinstance(module_, (torch.nn.LayerNorm)) or 'LayerNorm' in module_.__class__.__name__:
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params

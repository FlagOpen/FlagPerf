# Copyright (c) 2023 BAAI. All rights reserved.
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
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from transformers import TransfoXLConfig, TransfoXLModel
from transformers import TransfoXLTokenizer, TransfoXLForSequenceClassification
from transformers import TransfoXLLMHeadModel

from transformers import T5Model


def _model_to_ddp(model, config):
    if torch.distributed.is_initialized():
        return DistributedDataParallel(model,
                                       find_unused_parameters=True,
                                       device_ids=[config.local_rank])
    return model


def create_model(config, device):
    hfconfig = TransfoXLConfig(
        n_layer=16,
        d_model=410,
        d_embed=410,
        n_head=10,
        d_head=41,
        d_inner=2100,
        dropout=0.1,
        dropatt=0.0,
        mem_len=150,
    )
    model = TransfoXLLMHeadModel(hfconfig).to(device)
    model.post_init()
    # model = TransfoXLLMHeadModel.from_pretrained(config.data_dir+"/model").to(device)
    model = _model_to_ddp(model, config)
    tokenizer = TransfoXLTokenizer()
    return model, hfconfig, tokenizer

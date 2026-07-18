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

import os
from collections import namedtuple

from transformers import DistilBertConfig, DistilBertTokenizer
from transformers import DistilBertForSequenceClassification


def create_model(config):
    model_path = os.path.join(config.data_dir, 'model')
    hfconfig = DistilBertConfig.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path,
                                                       config=hfconfig)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    return model, hfconfig, tokenizer


if __name__ == '__main__':

    Config = namedtuple('Config', ['data_dir'])
    config = Config('distilbert')
    model, model_config, tokenizer = create_model(config)
    import pdb; pdb.set_trace()

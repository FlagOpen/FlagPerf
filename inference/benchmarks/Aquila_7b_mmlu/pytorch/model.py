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

import torch
from loguru import logger

from flagai.auto_model.auto_loader import AutoLoader


def create_model(config):
    logger.info("build model...")
    if os.path.exists(config.download_path):
        model_dict = torch.load(config.download_path)
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
    else:
        loader = AutoLoader(
        "lm",
        model_dir=config.state_dict,
        model_name=config.model_name,
        use_cache=True,
        fp16=True)
        model = loader.get_model()
        tokenizer = loader.get_tokenizer()
        torch.save({"model":model, "tokenizer":tokenizer}, config.download_path)

    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model

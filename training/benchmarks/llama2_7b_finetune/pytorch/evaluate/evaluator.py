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

from loguru import logger
import torch

from dataset import build_dataloader


def evaluator(pred, y):
    gt = float(y[0][0][1])
    predict = pred[:, -1, :]
    answer = float(torch.argmax(predict, dim=1))
    if answer == gt:
        return 1
    else:
        return 0


def evaluate_MMLU(model, config, tokenizer):
    dataloader = build_dataloader(config, tokenizer)
    token_cnt = 0
    correct = 0
    whole = 0

    for step, item in enumerate(dataloader):
        if step % config.log_freq == 0:
            logger.debug("Step: " + str(step) + " / " + str(len(dataloader)))

        tokens = item["prompt"].input_ids.cuda()[0]

        with torch.no_grad():
            y = model(tokens)
            token_cnt += len(tokens[0])
            pred = y[0]
            r = evaluator(pred, item["answer"])
            correct += r
            whole += 1

    logger.info("MMLU" + str(config.few_shots) + "-shots Acc: " +
                str(correct / whole))

    return round(correct / whole, 3)

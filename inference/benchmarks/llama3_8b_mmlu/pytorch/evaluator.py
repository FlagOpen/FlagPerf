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


def evaluator(pred, y, dataloader):

    tokenizer = dataloader.dataset.tokenizer

    gt = y[0][0][0]
    predict = pred[:, -1, :]
    answer = torch.argmax(predict, dim=1)
    answer_str = tokenizer.decode(answer)
    valid_answers = ['A', 'B', 'C', 'D']
    answer_str = ''.join([c for c in answer_str if c in valid_answers])
    gt_str = tokenizer.decode(gt)
    if answer_str == gt_str:
        return 1
    else:
        return 0

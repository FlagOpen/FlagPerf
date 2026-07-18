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


class Evaluator:
    """Evaluator"""

    def __init__(self):
        pass

    def accuracy(self, output, input_ids, labels):
        pred = torch.argmax(output.logits, dim=2)

        mask = input_ids == 103
        masked_pred = pred[mask]
        masked_label = labels[mask]
        correct = masked_pred[masked_pred == masked_label]

        return len(correct) / len(masked_label)

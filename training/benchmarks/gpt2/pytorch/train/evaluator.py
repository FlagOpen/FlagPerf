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
import sys
import torch

from train.utils import process_batch_eval

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver import dist_pytorch

class Evaluator:

    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader

    def evaluate(self, trainer):
        model = trainer.model
        model.eval()

        total_output = 0.0
        num_examples = len(self.eval_dataloader.dataset)
        with torch.no_grad():
            # For all the batches in the dataset.
            for iteration, batch in enumerate(self.eval_dataloader):

                # Get the batch.
                tokens, labels, attention_mask, position_ids, loss_mask = process_batch_eval(
                    batch)
                # Forward pass through the model.
                output = model(tokens, position_ids, attention_mask)
                # For accuracy, return the number of correctly predicted samples.
                outputs = torch.argmax(output, -1)
                correct = (outputs == labels).float()
                correct[(1 - loss_mask).bool()] = 1
                correct = correct.prod(-1)
                output = correct.sum()

                # Reduce across processes.
                if dist_pytorch.is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(output)

                total_output += output
        acc = total_output / num_examples
        model.eval()
        return acc.item()

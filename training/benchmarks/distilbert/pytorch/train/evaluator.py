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
from torch.types import Device

from driver import dist_pytorch

class Evaluator:
    """Evaluator"""
    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader
        self.device = config.device
    
    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch

    def evaluate(self, trainer):
        model = trainer.model
        model.eval()

        total_output = 0.0
        num_examples = len(self.eval_dataloader.dataset)
        with torch.no_grad():
            # For all the batches in the dataset.
            for step, inputs in enumerate(self.eval_dataloader):
                # Forward pass through the model.
                inputs = self.process_batch(inputs, self.device)
                output = model(**inputs)
                # For accuracy, return the number of correctly predicted samples.
                outputs = torch.argmax(output['logits'], -1)
                correct = (outputs == inputs['labels']).float()
                output = correct.sum()

                # Reduce across processes.
                if dist_pytorch.is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(output)

                total_output += output
        acc = total_output / num_examples
        return acc.item()

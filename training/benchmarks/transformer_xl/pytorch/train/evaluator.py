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
from torch.types import Device


class Evaluator:
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch

    def evaluate(self, trainer):
        model = trainer.model
        model.eval()

        nlls = []
        total_eval_loss = torch.tensor(0.0).cuda()
        total_examples = torch.tensor(0).cuda()

        with torch.no_grad():
            for batch in self.dataloader:
                batch = self.process_batch(batch, trainer.device)
                loss = model(**batch)
                total_eval_loss += loss.loss*loss.losses.shape[0]
                total_examples += loss.losses.shape[0]
                nlls.append(loss.loss.item())
                torch.cuda.synchronize()
        trainer.model.train()

        ppl_size = torch.tensor(len(nlls)).cuda()
        ppl_sum = torch.exp(torch.tensor(nlls)).sum().cuda()

        if torch.distributed.is_initialized():
            # Collect total scores from all ranks
            torch.distributed.all_reduce(
                total_eval_loss, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                total_examples, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                ppl_sum, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                ppl_size, op=torch.distributed.ReduceOp.SUM
            )
        # Average by number of examples
        total_eval_loss = total_eval_loss / total_examples
        ppl = ppl_sum/ppl_size
        return total_eval_loss.item(), ppl.item()

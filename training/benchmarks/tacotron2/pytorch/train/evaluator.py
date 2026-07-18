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
import time
from contextlib import contextmanager

from model.data.data_function import batch_to_gpu
from .utils import reduce_tensor

# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license


@contextmanager
def evaluating(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


class Evaluator:

    def __init__(self, config, val_dataloader):
        self.config = config
        self.val_dataloader = val_dataloader

    def evaluate(self, trainer):
        """Handles all the validation scoring"""
        model = trainer.model
        criterion = trainer.criterion
        world_size = trainer.world_size
        distributed_run = trainer.config.distributed

        with evaluating(model), torch.no_grad():
            val_loader = self.val_dataloader

            val_loss = 0.0
            num_iters = 0
            val_items_per_sec = 0.0

            for _, batch in enumerate(val_loader):
                torch.cuda.synchronize()
                iter_start_time = time.perf_counter()

                x, y, num_items = batch_to_gpu(batch)
                loss = trainer.adapter.calculate_loss(model, trainer.config, criterion, x, y)

                if distributed_run:
                    reduced_val_loss = reduce_tensor(loss.data,
                                                     world_size).item()
                    reduced_num_items = reduce_tensor(num_items.data, 1).item()
                else:
                    reduced_val_loss = loss.item()
                    reduced_num_items = num_items.item()
                val_loss += reduced_val_loss

                torch.cuda.synchronize()
                iter_stop_time = time.perf_counter()
                iter_time = iter_stop_time - iter_start_time

                items_per_sec = reduced_num_items / iter_time
                val_items_per_sec += items_per_sec
                num_iters += 1

            val_loss = val_loss / num_iters
            val_items_per_sec = val_items_per_sec / num_iters

            return val_loss, val_items_per_sec

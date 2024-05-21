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
from dataclasses import dataclass
from time import time

from driver import Driver
from model import create_model
from optimizer import create_optimizer
from torch.types import Device
from train.training_state import TrainingState

from .evaluator import Evaluator
from scheduler import create_scheduler

def _process_batch(batch, device: Device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@dataclass
class Trainer:
    driver: Driver = (None,)
    evaluator: Evaluator = (None,)
    state: TrainingState = (None,)
    device: Device = (None,)

    def init(self, config):
        self.model, self.model_config, self.tokenizer = create_model(config, self.device)
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.target_ppl = config.target_ppl
        self.max_epoch = config.max_epoch
        self.batch_size = config.train_batch_size

    def train_one_epoch(self, dataloader):
        state = self.state
        state.global_step = 0
        state.num_steps = len(dataloader)
        self.model.train()
        with state.epoch_train_guard(self.driver):
            no_eval_start = time()
            for i, data in enumerate(dataloader):
                state.global_step += 1
                state.num_trained_samples += self.batch_size
                data = _process_batch(data, self.device)
                self.train_one_step(data)

                state.noevaltime += time() - no_eval_start
                no_eval_start = time()
        with state.epoch_eval_guard(self.driver):
            state.loss, state.ppl = self.evaluator.evaluate(self)
        self.detect_training_status(state)

    def train_one_step(self, data):
        with self.state.step_guard(self.driver):
            outputs = self.model(**data)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
        self.scheduler.step()
        self.state.loss = loss.item()

    def detect_training_status(self, state: TrainingState):
        if state.ppl <= self.target_ppl:
            state.converged_success()

        if state.epoch >= self.max_epoch:
            state.end_training = True

        return state.end_training

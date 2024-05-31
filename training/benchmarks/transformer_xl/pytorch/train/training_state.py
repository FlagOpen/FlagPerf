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

import time
from contextlib import contextmanager
from dataclasses import dataclass

from driver import Driver, Event


@dataclass
class TrainingState:
    """TrainingState dataclass"""

    _status = "aborted"  # later set to "success" if termination criteria met

    epoch: int = 1
    global_step: int = 0
    num_steps: int = 0

    loss: float = 0.0
    ppl: float = None
    end_training: bool = False
    converged: bool = False

    traintime: float = 0.0
    noevaltime: float = 0.0
    purecomputetime: float = 0.0

    num_trained_samples: int = 0

    def status(self):
        """get status"""
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        """converged success"""
        self.end_training = True
        self.converged = True

    @contextmanager
    def epoch_train_guard(self, driver: Driver) -> None:
        driver.event(Event.EPOCH_BEGIN, self.epoch)
        yield
        driver.event(Event.EPOCH_END, self.epoch)

    @contextmanager
    def epoch_eval_guard(self, driver: Driver) -> None:
        eval_start = time.time()
        yield
        eval_end = time.time()
        eval_result = dict(
            global_step=self.global_step, ppl=self.ppl, loss=self.loss, time=eval_end - eval_start
        )
        driver.event(Event.EVALUATE, eval_result)

    @contextmanager
    def step_guard(self, driver: Driver) -> None:
        driver.event(Event.STEP_BEGIN, step=self.global_step)
        pure_compute_start = time.time()
        yield
        pure_compute_end = time.time()
        self.purecomputetime += pure_compute_end - pure_compute_start
        driver.event(
            Event.STEP_END,
            step=self.global_step,
            loss=self.loss,
            message=dict(step=f"{self.global_step}/{self.num_steps}",
                         loss=self.loss,
                         time=pure_compute_end - pure_compute_start)
        )

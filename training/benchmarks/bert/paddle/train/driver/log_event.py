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

import config
from .event import Event, EventManager
from .perf_logger import PerfLogger, LogLevel

import copy

STACKLEVEL = 4


class LogEventManager(EventManager):

    def __init__(self, logger: PerfLogger = None, log_freq: int = 0):
        super(LogEventManager, self).__init__()
        self.log_freq = log_freq
        level = LogLevel.INFO if log_freq > 0 else LogLevel.SUBMITTION
        self.logger = logger or PerfLogger.get_default_logger(
            rank=config.local_rank, level=level)

    def on_launch_training(self):
        self._log_event(Event.LAUNCH_TRAINING,
                        "Launch training",
                        stacklevel=STACKLEVEL)

    def on_init_evaluation(self, result: dict):
        self._log_event(Event.INIT_EVALUATION, result)

    def on_evaluate(self, result: dict):
        self._log_event(Event.EVALUATE, result)

    def on_backward(self, step: int, loss, optimizer, grad_scaler=None):
        pass
        #self._log_event(Event.BACKWARD, f"step: {step}  loss: {loss}")

    def on_init_start(self):
        self._log_event(Event.INIT_START)

    def on_init_end(self):
        self._log_event(Event.INIT_END, "Finish initialization")

    def on_train_start(self):
        self._log_event(Event.TRAIN_START)

    def on_train_end(self):
        self._log_event(Event.TRAIN_END)

    def on_epoch_begin(self, epoch: int):
        epoch_info = dict(epoch=epoch)
        self._log_event(Event.EPOCH_BEGIN, epoch_info)

    def on_epoch_end(self, epoch: int):
        epoch_info = dict(epoch=epoch)
        self._log_event(Event.EPOCH_END, epoch_info)

    def on_step_begin(self, step: int = None):
        pass

    def on_step_end(self, step: int = None, loss=None, message=None):
        if (self.log_freq <= 0 or step % self.log_freq != 0) and step != 1:
            return
        self._log_event(Event.STEP_END, message=message)

    def _log_event(self, event, *args, **kwargs):
        self.logger.log(event, stacklevel=STACKLEVEL, *args, **kwargs)

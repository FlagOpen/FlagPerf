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

from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    """TrainingState dataclass"""
    global_steps = 0

    loss: float = 0.0
    eval_mAP: float = 0.0

    epoch: int = 1
    end_training: bool = False
    converged: bool = False

    num_trained_samples: int = 0

    init_time = 0
    raw_train_time = 0
    no_eval_time = 0.0
    pure_compute_time = 0.0

    def converged_success(self):
        """converged success"""
        self.end_training = True
        self.converged = True

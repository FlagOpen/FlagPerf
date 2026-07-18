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
import torch._dynamo

import time


class InferModel:

    def __init__(self, config, onnx_path, model):
        self.config = config
        torch._dynamo.reset()
        self.model = torch.compile(model, mode=config.dynamo_mode, dynamic=config.dynamo_dynamic)
        self.warmup = config.dynamo_wamrup_iters

    def __call__(self, model_inputs: list):
        start = time.time()
        if self.warmup != 0:
            for i in range(self.config.dynamo_wamrup_times):
                _ = self.model(model_inputs[0])
            self.warmup -= 1

        torch.cuda.synchronize()
        compile_foo_time = time.time() - start

        model_outputs = self.model(model_inputs[0].cuda())
        return [model_outputs], compile_foo_time

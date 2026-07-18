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
import torch_tensorrt as torchtrt
import time


class InferModel:

    def __init__(self, config, onnx_path, model):
        self.config = config
        self.origin_model = model
        self.traced_model = None
        self.trt_model = None
        self.full_compile = config.torchtrt_full_compile

    def __call__(self, model_inputs: list):
        start = time.time()
        model_cuda_inputs = []
        for item in model_inputs:
            model_cuda_inputs.append(item.cuda())

        if self.traced_model is None:
            self.traced_model = torch.jit.trace(self.origin_model,
                                                model_cuda_inputs)
            self.trt_model = torchtrt.compile(
                self.traced_model,
                inputs=model_cuda_inputs,
                truncate_long_and_double=True,
                enabled_precisions={torch.float32, torch.float16},
                require_full_compilation=self.full_compile)

        compile_foo_time = time.time() - start

        model_outputs = self.trt_model(*model_cuda_inputs)
        return [model_outputs], compile_foo_time

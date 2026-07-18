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

import onnxruntime
import torch
import numpy as np

class InferModel:

    def __init__(self, config, onnx_path, model):
        self.config = config
        self.engine = self.build_engine(config, onnx_path)
        self.input_names = self.engine.get_inputs()[0].name
        self.output_names = [x.name for x in self.engine.get_outputs()]

    def build_engine(self, config, onnx_path):
        providers = ["MACAExecutionProvider"]
        provider_options = [{"device_id": 0}]
        session = onnxruntime.InferenceSession(onnx_path, providers=providers, provider_options=provider_options)
        return session

    def __call__(self, model_inputs: list):
        pred_out_list = self.engine.run(self.output_names, {self.input_names:model_inputs[0].numpy()})
        result = torch.from_numpy(np.array(pred_out_list))
        return result, 0.0

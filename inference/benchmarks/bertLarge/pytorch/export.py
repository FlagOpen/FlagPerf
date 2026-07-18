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
import os


def export_model(model, config):
    if config.exist_onnx_path is not None:
        return config.exist_onnx_path

    filename = config.case + "_bs" + str(config.batch_size)
    filename = filename + "_" + str(config.framework)
    filename = filename + "_fp16" + str(config.fp16)
    filename = "onnxs/" + filename + ".onnx"
    onnx_path = config.perf_dir + "/" + filename

    dummy_input = torch.ones(config.batch_size, config.seq_length).int().cuda()

    dir_onnx_path = os.path.dirname(onnx_path)
    os.makedirs(dir_onnx_path, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          verbose=False,
                          input_names=["input"],
                          output_names=["output"],
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True)

    return onnx_path

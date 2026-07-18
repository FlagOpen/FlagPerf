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

    latent = torch.randn(config.batch_size * 2, config.in_channels,
                         config.height // config.scale_size,
                         config.width // config.scale_size).cuda().float()
    t = torch.randn([]).cuda().int()
    embed = torch.randn(config.batch_size * 2, config.prompt_max_len,
                        config.embed_hidden_size).cuda().float()

    if config.fp16:
        latent = latent.half()
        embed = embed.half()

    dummy_input = (latent, t, embed)

    dir_onnx_path = os.path.dirname(onnx_path)
    os.makedirs(dir_onnx_path, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          verbose=False,
                          input_names=["input_0", "input_1", "input_2"],
                          output_names=["output_0"],
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True)

    return onnx_path

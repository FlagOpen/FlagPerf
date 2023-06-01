# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import torch.nn as nn


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
<<<<<<< HEAD
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"
=======
        assert (module.weight.size(1) % block_size == 0
                ), "Input features must be a multiple of block sizes"
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
<<<<<<< HEAD
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
=======
            assert (module.in_channels % block_size == 0
                    ), "Input channels must be a multiple of block sizes"
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
<<<<<<< HEAD
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
=======
                mask = torch.zeros(in_features // block_size * out_features,
                                   device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size,
                                              -1).view(-1, in_features)
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
<<<<<<< HEAD
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )
=======
                    mask = mask.repeat_interleave(block_size,
                                                  -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0),
                                       weight.size(1),
                                       device=weight.device)
                    mask.bernoulli_(p)
                    mask = (mask.unsqueeze(2).unsqueeze(3).repeat(
                        1, 1, mod.kernel_size[0], mod.kernel_size[1]))
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

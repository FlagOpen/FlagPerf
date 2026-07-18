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
import swin_window_process


class WindowProcess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = swin_window_process.roll_and_window_partition_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size
        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        grad_out = swin_window_process.roll_and_window_partition_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = swin_window_process.window_merge_and_roll_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size

        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        grad_out = swin_window_process.window_merge_and_roll_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None

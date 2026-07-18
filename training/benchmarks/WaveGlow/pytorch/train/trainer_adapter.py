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
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from driver import dist_pytorch


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../../")))
from driver.dist_pytorch import main_proc_print


def convert_model(model):
    """convert_model"""
    return model


def model_to_fp16(model, config):
    """model_to_fp16"""
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model, config):
    """model_to_ddp"""
    if dist_pytorch.is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[config.local_rank])
    return model


def create_grad_scaler(args):
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    return scaler

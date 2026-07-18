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
from apex.optimizers import FusedAdam as Adam
from apex.contrib.clip_grad import clip_grad_norm_

def create_optimizer(model, args, loss_module):
    optimizer = Adam((*model.parameters(), *loss_module.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
    return optimizer


def create_clip_grad():
    return clip_grad_norm_


def create_grad_scaler(args):
    """create_grad_scaler for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    return scaler

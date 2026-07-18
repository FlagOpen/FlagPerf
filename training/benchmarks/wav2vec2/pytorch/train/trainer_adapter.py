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

import torch.distributed as dist
import config

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
      
from common.fairseq.optim.fused_adam import get_fused_adam_class
from driver.dist_pytorch import main_proc_print


def convert_model(model: nn.Module) -> nn.Module:
    return model


def model_to_fp16(model):
    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if config.fp16:
        main_proc_print(" > use fp16...")
        model.half()
    return model


def model_to_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        model = DDP(model,
                    device_ids=[config.local_rank],
                    find_unused_parameters=True)
        from common.fairseq.dist import ModuleProxyWrapper
        model = ModuleProxyWrapper(model)
    return model


def create_optimizer(model, args):

    kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'fused_adam' and not (args.fp16 or args.bf16):
        kw.update({'betas': args.adam_betas, 'eps': args.adam_eps})
        fused_adam_cls = get_fused_adam_class()
        print(fused_adam_cls, "fused_adam_cls")
        optimizer = fused_adam_cls(model.parameters(), **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    return optimizer

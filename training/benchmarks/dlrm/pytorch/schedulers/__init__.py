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

from driver import dist_pytorch
from utils.utils import LearningRateScheduler


def create_scheduler(args, mlp_optimizer, embedding_optimizer):
    data_parallel_lr = args.lr
    world_size = args.n_device

    if args.Adam_MLP_optimizer:
        MLP_model_parallel_lr = args.lr
    else:
        MLP_model_parallel_lr = args.lr / world_size

    if dist_pytorch.is_main_process():
        mlp_lrs = [data_parallel_lr, MLP_model_parallel_lr]
    else:
        mlp_lrs = [data_parallel_lr]

    # DDP introduces a gradient average through allreduce(mean), which doesn't apply to bottom model.
    # Compensate it with further scaling lr
    if args.Adam_embedding_optimizer:
        embedding_model_parallel_lr = args.lr
    else:
        embedding_model_parallel_lr = args.lr / world_size

    embedding_lrs = [embedding_model_parallel_lr]

    lr_scheduler = LearningRateScheduler(
        optimizers=[mlp_optimizer, embedding_optimizer],
        base_lrs=[mlp_lrs, embedding_lrs],
        warmup_steps=args.warmup_steps,
        warmup_factor=args.warmup_factor,
        decay_start_step=args.decay_start_step,
        decay_steps=args.decay_steps,
        decay_power=args.decay_power,
        end_lr_factor=args.decay_end_lr / args.lr)

    return lr_scheduler

# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""
A simple launcher script for TX8 training

Inspired by https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

::
    >>> python xla_spawn.py YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

"""
import os
import functools
import importlib
import sys
from argparse import REMAINDER, ArgumentParser
from multiprocessing import Semaphore, Manager
from pathlib import Path
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_env_vars as xenv


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "PyTorch TX8 distributed training launch helper utility that will spawn up multiple distributed processes"
        )
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the single TX8 training "
            "program/script to be launched in parallel, "
            "followed by all the arguments for the "
            "training script"
        ),
    )

    parser.add_argument("--fsdp_dp_sharding", type=int, default=-1, help="FSDP dp number..")

    parser.add_argument("--megatron_tp_sharding", type=int, default=1, help="megatron tp number.")

    parser.add_argument("--semaphore_number", type=int, default=0, help="semaphore number.")

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser.parse_args()

def main():
    args = parse_args()

    mg_semap = None
    if args.semaphore_number > 0:
        manager = Manager()
        mg_semap = manager.Semaphore(args.semaphore_number)
    # Import training_script as a module.
    script_fpath = Path(args.training_script)
    sys.path.append(str(script_fpath.parent.resolve()))
    mod_name = script_fpath.stem
    mod = importlib.import_module(mod_name)
    local_world_size = os.environ.get(xenv.PJRT_LOCAL_WORLD_SIZE, None)
    if local_world_size is None:
        raise RuntimeError("PJRT_LOCAL_WORLD_SIZE cannot be None!")

    # Patch sys.argv
    sys.argv = [args.training_script] + args.training_script_args + \
        ["--fsdp_dp_sharding", str(args.fsdp_dp_sharding), "--megatron_tp_sharding", str(args.megatron_tp_sharding)]

    device_preprocess = functools.partial(mod._mp_device_preprocess, fsdp_dp_sharding=args.fsdp_dp_sharding, megatron_tp_sharding=args.megatron_tp_sharding)
    xmp.spawn(mod._mp_fn, device_preprocess=mod._mp_device_preprocess, args=(mg_semap,), nprocs=int(local_world_size))


if __name__ == "__main__":
    main()

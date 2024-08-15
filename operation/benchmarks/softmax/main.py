# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import os
import time
from argparse import ArgumentParser, Namespace
import yaml
import sys
import subprocess
import math

sys.path.append("..")
from drivers.utils import *
from drivers.calculate import *


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")
    parser.add_argument("--case_name",
                        type=str,
                        required=True,
                        help="op name like mm")
    parser.add_argument("--spectflops",
                        type=str,
                        required=True,
                        help="spectflops of current dataformat")

    parser.add_argument("--dataformat",
                        type=str,
                        required=True,
                        help="like FP32,FP16")

    parser.add_argument("--oplib",
                        type=str,
                        required=True,
                        help="impl like pytorch/flaggems/cpp")

    parser.add_argument("--chip",
                        type=str,
                        required=True,
                        help="chip like A100_40_SXM")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def main(config, case_config):
    correctness = do_correctness(config.case_name)
    correctness = correctness == 0
    dtype = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
        "INT32": torch.int32,
        "INT16": torch.int16,
        "BOOL": torch.bool
        }
    set_ieee_float32(config.vendor)

    Melements = case_config.Melements
    # default shape: (M, 1024, 1024)
    shape = (Melements, 1024, 1024)

    if config.vendor == 'kunlunxin':
        # if `Shape' specified in `case_config.yaml', use it
        if case_config.__contains__('Shape') and case_config.Shape is not None:
            shape = case_config.Shape

    a = torch.randn(shape, dtype=dtype[config.dataformat], requires_grad=True).to(0)
    print(f'Shape for performance_test: {a.shape}')

    f = torch.nn.Softmax(dim=1).to(0)

    latency_nowarm, latency_warm, cputime, kerneltime = do_test(
        f, (a, ), host_device_sync, config, case_config, bp=True)

    op2flops = lambda x: x * 3 * math.prod(shape)

    perf_result = cal_perf(cputime, kerneltime, op2flops,
                           config.spectflops, bp=True)
    print_result(config, config.case_name, *perf_result, correctness,
                 latency_nowarm, latency_warm)


if __name__ == "__main__":
    config = parse_args()
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)
    adapt_torch(config.vendor)
    with open(os.path.join(config.vendor, config.chip, "case_config.yaml"),
              "r") as file:
        case_config_vendor = yaml.safe_load(file)
    case_config.update(case_config_vendor)
    case_config = Namespace(**case_config)

    if config.oplib == "flaggems":
        import flag_gems
        flag_gems.enable()
        print("Using flaggems")
    else:
        print("Using nativetorch")
    main(config, case_config)

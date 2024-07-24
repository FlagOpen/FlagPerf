 # Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import torch.distributed as dist
import os
import time
from argparse import ArgumentParser, Namespace
import yaml
import sys

sys.path.append("..")
from drivers.utils import *
from drivers.calculate import *


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

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
    set_ieee_float32(config.vendor)

    print("Test Correctness with 1M-times smaller operation")
    m = case_config.Melements
    f = torch.nn.SiLU()
    dtype = {"FP32": torch.float32}

    mmape = []

    torch.manual_seed(42)
    for i in range(100):
        a = torch.randn(m, dtype=dtype[config.dataformat])

        a_fp64 = a.to(torch.float64)
        r_fp64 = f(a_fp64) # 修改为gelu

        a = a.to(0)
        r_device = f(a).cpu() 
        mape = torch.mean(torch.abs(r_device - r_fp64) / torch.abs(r_fp64))

        mmape.append(mape)
    
    mape = torch.mean(torch.tensor(mmape))
    mape_std = torch.std(torch.tensor(mmape))

    a = torch.randn(m, 1024, 1024, dtype=dtype[config.dataformat]).to(0)

    latency_nowarm, latency_warm, cputime, kerneltime = do_test(
        f, (a, ), host_device_sync, config, case_config) # 调整为torch.sub

    op2flops = lambda x: x * 9 * m * 1024 * 1024 # 根据减法的实际FLOPs调整

    perf_result = cal_perf(cputime, kerneltime, op2flops,
                           case_config.SPECTFLOPS)
    print_result(config, "silu", *perf_result, mape, mape_std,
                 latency_nowarm, latency_warm)


if __name__ == "__main__":
    config = parse_args()
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)
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
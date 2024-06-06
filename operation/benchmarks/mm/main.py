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
import triton
import sys

sys.path.append("..")
from drivers.utils import *


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
    print("Test Correctness with 16-times smaller operation")

    m = case_config.M
    n = case_config.N
    k = case_config.K

    dtype = {"FP16": torch.float16}

    mmape = []

    torch.manual_seed(42)
    for i in range(100):
        a = torch.randn((m // 16, n // 16), dtype=dtype[config.dataformat])
        b = torch.randn((n // 16, k // 16), dtype=dtype[config.dataformat])

        a_fp64 = a.to(torch.float64)
        b_fp64 = b.to(torch.float64)
        r_fp64 = torch.mm(a, b)

        a = a.to(0)
        b = b.to(0)

        r_device = torch.mm(a, b).cpu()
        mape = torch.mean(torch.abs(r_device - r_fp64) / torch.abs(r_fp64))
        mmape.append(mape)
    mape = torch.mean(torch.tensor(mmape))
    mape_std = torch.std(torch.tensor(mmape))

    a = torch.randn((m, n), dtype=dtype[config.dataformat]).to(0)
    b = torch.randn((n, k), dtype=dtype[config.dataformat]).to(0)

    host_device_sync(config.vendor)
    start_latency_nowarm = time.perf_counter_ns()
    _tensor = torch.mm(a, b)

    host_device_sync(config.vendor)
    latency_nowarm = time.perf_counter_ns() - start_latency_nowarm

    print("start warmup")

    for _ in range(case_config.WARMUP):
        _tensor = torch.mm(a, b)

    host_device_sync(config.vendor)
    start_latency_warm = time.perf_counter_ns()
    _tensor = torch.mm(a, b)

    host_device_sync(config.vendor)
    latency_warm = time.perf_counter_ns() - start_latency_warm

    start_time = time.perf_counter()
    for _ in range(case_config.ITERS):
        _tensor = torch.mm(a, b)

    host_device_sync(config.vendor)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    datasize = case_config.ITERS * (m * n * k * 2)
    tflops = datasize / elapsed_time / 1E12

    kernel_latency = triton.testing.do_bench(lambda: torch.mm(a, b),
                                             warmup=case_config.KERNELWARMUP,
                                             rep=case_config.KERNELITERS,
                                             return_mode="median")
    kernel_tflops = round(2 * m * n * k / (kernel_latency / 1000.0) / 1E12, 2)
    ct = elapsed_time / case_config.ITERS
    kt = kernel_latency / 1000.0
    cps = 1.0 / ct
    kps = 1.0 / kt
    cflops = 2 * m * n * k * cps
    kflops = 2 * m * n * k * kps

    cfu = round(100.0 * cflops / 1E12 / case_config.SPECTFLOPS, 2)
    kfu = round(100.0 * kflops / 1E12 / case_config.SPECTFLOPS, 2)
    return round(ct * 1E6, 2), round(kt * 1E6, 2), round(cps, 2), round(
        kps, 2), round(cflops / 1E12,
                       2), round(kflops / 1E12, 2), mape, mape_std, round(
                           latency_nowarm / 1000.0,
                           2), round(latency_warm / 1000.0, 2), cfu, kfu 


if __name__ == "__main__":
    config = parse_args()
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)
    with open(os.path.join(config.vendor, config.chip, "case_config.yaml"),
              "r") as file:
        case_config_vendor = yaml.safe_load(file)
    case_config.update(case_config_vendor)
    case_config = Namespace(**case_config)

    print(case_config)
    if config.oplib == "flaggems":
        import flag_gems
        flag_gems.enable()
        print("Using flaggems")
    else:
        print("Using nativetorch")
    ct, kt, cps, kps, ctflops, ktflops, errmean, errstd, lnm, lm, cfu, kfu = main(
        config, case_config)
    print(
        r"[FlagPerf Result]Operation matrix multiply(mm) in {} at {}:".format(
            config.oplib, config.dataformat))
    print(r"[FlagPerf Result]FLOPS utilization: cputime={}%, kerneltime={}%".format(cfu, kfu))
    print(
        r"[FlagPerf Result]cputime={} us, throughput={} op/s, equals to {} TFLOPS"
        .format(ct, cps, ctflops))
    print(
        r"[FlagPerf Result]kerneltime={} us, throughput={} op/s, equals to {} TFLOPS"
        .format(kt, kps, ktflops))
    print(r"[FlagPerf Result]Relative error with FP64-CPU: mean={}, std={}".
          format(errmean, errstd))
    print(
        r"[FlagPerf Result]First time latency: no warmup={} us, warmup={} us".
        format(lnm, lm))

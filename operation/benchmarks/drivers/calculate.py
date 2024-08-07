# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from triton.testing import do_bench as kernel_bench
import os
import subprocess


def do_correctness(operation):
    gems_repo = subprocess.check_output(
        ["find", "/", "-type", "d", "-name", "FlagGems"], text=True).strip()

    p = subprocess.Popen(
        f"cd {os.path.join(gems_repo, 'tests')} && python3 test_specific_ops.py --name {operation}",
        shell=True
        )
    p.wait()

    return p.returncode


def do_test(exec_func, exec_args, sync_func, config, case_config):
    sync_func(config.vendor)
    start_latency_nowarm = time.perf_counter_ns()
    _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    latency_nowarm = time.perf_counter_ns() - start_latency_nowarm

    for _ in range(case_config.WARMUP):
        _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    start_latency_warm = time.perf_counter_ns()
    _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    latency_warm = time.perf_counter_ns() - start_latency_warm

    start_time = time.perf_counter()
    for _ in range(case_config.ITERS):
        _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    end_time = time.perf_counter()

    cputime_raw = end_time - start_time

    kerneltime_raw = kernel_bench(lambda: exec_func(*exec_args),
                                  warmup=case_config.KERNELWARMUP,
                                  rep=case_config.KERNELITERS,
                                  return_mode="median")
    cputime = cputime_raw / case_config.ITERS
    kerneltime = kerneltime_raw / 1000.0  # ms to s
    return round(latency_nowarm / 1000.0, 2), round(latency_warm / 1000.0,
                                                    2), cputime, kerneltime


def cal_perf(cputime, kerneltime, op2flops, spectflops):
    spectflops = float(spectflops)
    ctus = round(cputime * 1E6, 2)
    ktus = round(kerneltime * 1E6, 2)

    cps = 1.0 / cputime
    kps = 1.0 / kerneltime

    cflops = op2flops(cps)
    kflops = op2flops(kps)
    ctflops = round(cflops / 1E12, 2)
    ktflops = round(kflops / 1E12, 2)

    cfu = round(100.0 * cflops / 1E12 / spectflops, 2)
    kfu = round(100.0 * kflops / 1E12 / spectflops, 2)

    return ctus, ktus, cps, kps, ctflops, ktflops, cfu, kfu


def print_result(config, casename, ct, kt, cps, kps, ctflops, ktflops, cfu,
                 kfu, correctness, lnm, lm):
    print(r"[FlagPerf Result]Operation {} in {} at {}:".format(
        casename, config.oplib, config.dataformat))
    print(r"[FlagPerf Result]FLOPS utilization: cputime={}%, kerneltime={}%".
          format(cfu, kfu))
    print(
        r"[FlagPerf Result]cputime={} us, throughput={} op/s, equals to {} TFLOPS"
        .format(ct, cps, ctflops))
    print(
        r"[FlagPerf Result]kerneltime={} us, throughput={} op/s, equals to {} TFLOPS"
        .format(kt, kps, ktflops))
    print(r"[FlagPerf Result]Correctness with CPU golden Reference: {}".format(
        correctness))
    print(
        r"[FlagPerf Result]First time latency: no warmup={} us, warmup={} us".
        format(lnm, lm))

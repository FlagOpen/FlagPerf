# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from loguru import logger
from triton.testing import do_bench as kernel_bench
import os
import subprocess


# test operation correctness
def do_correctness(operation):
    flaggems_dir = os.getenv("FLAGGEMS_WORK_DIR", "/")
    gems_repo = subprocess.check_output(
        ["find", flaggems_dir, "-type", "d", "-name", "FlagGems"], text=True).strip()
    p = subprocess.Popen(
        f"cd {os.path.join(gems_repo, 'tests')} && python3 test_named_ops.py --name {operation} --device cpu ",
        shell=True
        )
    p.wait()

    return p.returncode


# test operation performance
def do_performance(mode, warmup, result_log_dir, image_name):
    flaggems_dir = os.getenv("FLAGGEMS_WORK_DIR", "/")
    gems_repo = subprocess.check_output(
        ["find", flaggems_dir, "-type", "d", "-name", "FlagGems"], text=True).strip()
    del_file_path = os.path.join(gems_repo, 'benchmark')
    # 删除历史日志
    # del_file = os.path.join(del_file_path, f"result--level_core--mode_{mode}--warmup_{warmup}--record_log.log")
    del_file = os.path.join(del_file_path, f"test_distribution_perf--level_core--mode_{mode}--warmup_{warmup}--record_log.log")
    logger.info(del_file)
    del_process = subprocess.Popen(["rm", del_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(del_process)
    del_process.communicate()
    p = subprocess.Popen(
        # 执行所有算子
        # f"cd {os.path.join(gems_repo, 'benchmark')} && pytest --level core --mode {mode} --warmup {warmup} --record log",
        # 执行单个算子
        # f"cd {os.path.join(gems_repo, 'benchmark')} && pytest -m mm --level core --mode {mode} --warmup {warmup} --record log -s",
        # 执行文件
        f"cd {os.path.join(gems_repo, 'benchmark')} && pytest test_distribution_perf.py --level core --mode {mode} --warmup {warmup} --record log",
        shell=True
    )
    p.wait()
    # log_dir = os.path.join(gems_repo, "benchmark", "result--level_core--record_log")
    log_dir = os.path.join(gems_repo, "benchmark",
                           f"test_distribution_perf--level_core--mode_{mode}--warmup_{warmup}--record_log.log")

    cp_subprocess = subprocess.run(
        [f"docker cp {image_name}:{log_dir} {result_log_dir}/result.log.txt"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    cp_subprocess.communicate()
    return p.returncode
    # log_dir = os.path.join(gems_repo, "benchmark", f"result--level_core--mode_{mode}--warmup_{warmup}--record_log.log")
    # save_log_path = os.path.join(result_log_dir, "result.log.txt")
    # logger.info("======print do_performance save_log_path============")
    # logger.info(save_log_path)
    # with open(log_dir, "r", encoding="utf-8") as file_r, open(save_log_path, "w", encoding="utf-8") as file_w:
    #     for line in file_r:
    #         file_w.write(line + '\n')
    # return log_dir


grad_outputs = None


def do(exec_func, exec_args, bp=False):
    global grad_outputs
    if bp:
        import torch
        _tensor = exec_func(*exec_args).sum()
        if grad_outputs is None:
            grad_outputs = torch.zeros_like(_tensor)
        inputs = list(filter(lambda x: x.requires_grad, [*exec_args]))
        _grad = torch.autograd.grad(outputs=_tensor, inputs=inputs, grad_outputs=grad_outputs)
    else:
        _tensor = exec_func(*exec_args)


def do_test(exec_func, exec_args, sync_func, config, case_config, bp=False):
    sync_func(config.vendor)
    start_latency_nowarm = time.perf_counter_ns()
    _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    latency_nowarm = time.perf_counter_ns() - start_latency_nowarm

    for _ in range(case_config.WARMUP):
        do(exec_func, exec_args, bp)

    sync_func(config.vendor)
    start_latency_warm = time.perf_counter_ns()
    _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    latency_warm = time.perf_counter_ns() - start_latency_warm

    start_time = time.perf_counter()
    for _ in range(case_config.ITERS):
        do(exec_func, exec_args, bp)

    sync_func(config.vendor)
    end_time = time.perf_counter()

    cputime_raw = end_time - start_time

    kerneltime_raw = kernel_bench(lambda: do(exec_func, exec_args, bp),
                                  warmup=case_config.KERNELWARMUP,
                                  rep=case_config.KERNELITERS,
                                  return_mode="median")
    cputime = cputime_raw / case_config.ITERS
    kerneltime = kerneltime_raw / 1000.0  # ms to s
    return round(latency_nowarm / 1000.0, 2), round(latency_warm / 1000.0,
                                                    2), cputime, kerneltime


def cal_perf(cputime, kerneltime, op2flops, spectflops, bp=False):
    spectflops = float(spectflops)
    ctus = round(cputime * 1E6, 2)
    ktus = round(kerneltime * 1E6, 2)

    cps = 1.0 / cputime
    kps = 1.0 / kerneltime

    cflops = op2flops(cps) * (3.0 if bp else 1.0)
    kflops = op2flops(kps) * (3.0 if bp else 1.0)
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

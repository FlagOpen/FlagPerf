#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Vector 算力测试 - 测量向量运算性能
使用逐元素操作（element-wise operations）测量 Vector ALU 吞吐量
"""

import torch
import torch.distributed as dist
import os
import time
from argparse import ArgumentParser, Namespace
import yaml
import sys
sys.path.append("..")
from drivers.utils import *


def parse_args():
    parser = ArgumentParser(description="Vector Computation Benchmark")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    parser.add_argument("--node_size",
                        type=int,
                        required=True,
                        help="number of GPUs per node")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def main(config, case_config, rank, world_size, local_rank):
    """
    Vector 算力测试主函数

    测试方法：
    1. 创建大型向量
    2. 执行向量加法、乘法等逐元素操作
    3. 计算 Vector TOPS (Tera Operations Per Second)
    """

    if rank == 0:
        print("Finish initialization for Vector computation test")

    if "iluvatar" in config.vendor:
        torch.cuda.set_device(local_rank)

    # 向量大小配置
    vector_size = case_config.VECTOR_SIZE  # 例如: 256M elements
    dtype = torch.float32  # Vector 操作通常用 FP32

    # 创建测试向量
    vec_a = torch.randn(vector_size, dtype=dtype).to(local_rank)
    vec_b = torch.randn(vector_size, dtype=dtype).to(local_rank)
    vec_c = torch.randn(vector_size, dtype=dtype).to(local_rank)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)

    if rank == 0:
        print("Start warmup")

    # 预热阶段
    for _ in range(case_config.WARMUP):
        # Vector 操作: 融合乘加 (FMA)
        # result = a * b + c
        _result = vec_a * vec_b + vec_c

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)

    if rank == 0:
        print("Start test")

    # 正式测试
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    start_time = time.perf_counter()

    for _ in range(case_config.ITERS):
        # Vector FMA: 每个元素 2 次操作 (1 mul + 1 add)
        _result = vec_a * vec_b + vec_c

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()

    exec_time = end_time - start_time

    # 计算 Vector TOPS
    # 每次迭代: vector_size 个元素 × 2 次操作 (mul + add)
    operations = case_config.ITERS * vector_size * 2
    tops = operations / exec_time / 1e12  # TOPS

    return round(tops, 2)


if __name__ == "__main__":
    config = parse_args()

    # 加载通用配置
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)

    # 加载厂商特定配置（覆盖通用配置）
    vendor_config_path = os.path.join(config.vendor, "case_config.yaml")
    if os.path.exists(vendor_config_path):
        with open(vendor_config_path, "r") as file:
            case_config_vendor = yaml.safe_load(file)
        case_config.update(case_config_vendor)

    case_config = Namespace(**case_config)

    # 初始化分布式环境
    dist.init_process_group(backend=case_config.DIST_BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % config.node_size

    # 运行测试
    result = main(config, case_config, rank, world_size, local_rank)

    # 输出结果
    multi_device_sync(config.vendor)
    for output_rank in range(config.node_size):
        if local_rank == output_rank:
            print(f"[FlagPerf Result]Rank {dist.get_rank()}'s computation-VECTOR={result}TOPS")

            # 特殊处理：某些芯片有多个die
            if "iluvatar" in config.vendor:
                print(f"[FlagPerf Result]Rank {dist.get_rank()} BI-V150 has 2 chips "
                      f"and overall GPU computation-VECTOR={result*2}TOPS")
        multi_device_sync(config.vendor)

    dist.destroy_process_group()

# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# cambricon mlu import
try:
    from torch_mlu.utils.model_transfer import transfer
except ImportError:
    pass

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
    parser = ArgumentParser(description=" ")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")
    
    parser.add_argument("--node_size",
                        type=int,
                        required=True,
                        help="for pytorch")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args
    

def main(config, case_config, rank, world_size, local_rank):    
    if rank == 0:
        print("finish initialization")
    
    if "iluvatar" in config.vendor:
        torch.cuda.set_device(local_rank)
        
    m = case_config.M
    n = case_config.N
    k = case_config.K
    
    
    matrixA = torch.randn(m, n, dtype=torch.bfloat16).to(local_rank)
    matrixB = torch.randn(n, k, dtype=torch.bfloat16).to(local_rank)
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")
    
    for _ in range(case_config.WARMUP):
        _result = torch.mm(matrixA, matrixB)
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start test")
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    start_time = time.perf_counter()
    
    for _ in range(case_config.ITERS):
        _result = torch.mm(matrixA, matrixB)
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()
    
    exec_time = end_time - start_time
    
    operations = case_config.ITERS * 2 * m * n * k
    tflops = operations / exec_time / 1e12
    
    return round(tflops, 2)


if __name__ == "__main__":    
    config = parse_args()
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)
    with open(os.path.join(config.vendor, "case_config.yaml"), "r") as file:
        case_config_vendor = yaml.safe_load(file)
    case_config.update(case_config_vendor)
    case_config = Namespace(**case_config)
        
    dist.init_process_group(backend=case_config.DIST_BACKEND)  
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % config.node_size
      
    result = main(config, case_config, rank, world_size, local_rank)
    
    multi_device_sync(config.vendor)
    for output_rank in range(config.node_size):
        if local_rank == output_rank:
            print(r"[FlagPerf Result]Rank {}'s computation-BF16=".format(dist.get_rank()) + str(result) + "TFLOPS")
            if "iluvatar" in config.vendor:
                print(r"[FlagPerf Result]Rank {} BI-V150 has 2 chips and overall GPU computation-BF16=".format(dist.get_rank()) + str(result*2) + "TFLOPS")
        multi_device_sync(config.vendor)
        
    dist.destroy_process_group()


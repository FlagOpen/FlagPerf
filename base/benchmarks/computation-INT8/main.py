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

# mthreads torch_musa import
try:
    import torch_musa
except ImportError:
    pass

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
        
    m = case_config.M
    n = case_config.N
    k = case_config.K
    
    matrixA = torch.ones(m, n, dtype=torch.int8).to(local_rank)
    matrixB = torch.ones(n, k, dtype=torch.int8).to(local_rank)

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
    tops = operations / exec_time / 1e12
    
    return round(tops, 2)


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
            print(r"[FlagPerf Result]Rank {}'s computation-INT8=".format(dist.get_rank()) + str(result) + "TOPS")
            if "iluvatar" in config.vendor:
                print(r"[FlagPerf Result]Rank {} BI-V150 has 2 chips and overall GPU computation-INT8=".format(dist.get_rank()) + str(result*2) + "TOPS")
        multi_device_sync(config.vendor)
        
    dist.destroy_process_group()


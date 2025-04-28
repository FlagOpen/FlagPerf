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

E4M3MAX = 448
fp8max = E4M3MAX
def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia/A100")
    
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
    
    if torch.musa.is_available():
        torch.musa.set_device(local_rank)
        device = torch.device(f"musa")
        matrixA = torch.randn(m, n, dtype=torch.float32, device=device)
        matrixB = torch.randn(n, k, dtype=torch.float32, device=device)
    else:
        matrixA = torch.randn(m, n, dtype=torch.float32).to(local_rank)
        matrixB = torch.randn(n, k, dtype=torch.float32).to(local_rank)
    
    # get f8 tensor from inputs
    scale_a = matrixA.abs().max() / fp8max
    scale_b = matrixB.abs().max() / fp8max
    f8_a = (matrixA / scale_a).to(torch.float8_e4m3fn)
    f8_b = (matrixB / scale_b).to(torch.float8_e4m3fn)

    
    # fp32 golden result
    golden = torch.mm(f8_a.float(), f8_b.float()) * scale_a * scale_b
    # out_dtype scaled_mm result
    scale_out = golden.abs().max() / fp8max

    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")
    
    for _ in range(case_config.WARMUP):
        _result = torch._scaled_mm(
            f8_a, 
            f8_b, 
            scale_a=scale_a,
            scale_b=scale_b,
            scale_result=scale_out,
            out_dtype=torch.float8_e4m3fn,
            )
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start test")
    
    start_time = time.perf_counter()
    
    for _ in range(case_config.ITERS):
        _result = torch._scaled_mm(
            f8_a, 
            f8_b, 
            scale_a=scale_a,
            scale_b=scale_b,
            scale_result=scale_out,
            out_dtype=torch.float8_e4m3fn,
            )
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()
    
    exec_time = end_time - start_time
    print(f"[mthreads debug]:Rank {local_rank}'s exec time is {exec_time}")
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
            print(r"[FlagPerf Result]Rank {}'s computation-FP8=".format(dist.get_rank()) + str(result) + "TFLOPS")
        multi_device_sync(config.vendor)
        
    dist.destroy_process_group()



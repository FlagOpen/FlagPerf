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
import random
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
    set_ieee_float32(config.vendor)
    if rank == 0:
        print("finish initialization")
        
    if "iluvatar" in config.vendor:
        torch.cuda.set_device(local_rank)
        rank_dst = 16
    else:
        rank_dst = 8
    
    Melements = case_config.Melements
    torchsize = (Melements, 1024, 1024)
    if "mthreads" in config.vendor:
        tensor = torch.rand(torchsize, dtype=torch.float32).to(local_rank)
    else:
        tensor = torch.rand(torchsize, dtype=torch.float32).cuda()

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")

    for _ in range(case_config.WARMUP):
        if rank == 0:
            dist.send(tensor, dst=rank_dst)
        elif rank == rank_dst:
            dist.recv(tensor, src=0)
        
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    start_time = time.perf_counter()

    for _ in range(case_config.ITERS):
        if rank == 0:
            dist.send(tensor, dst=rank_dst)
        elif rank == rank_dst:
            dist.recv(tensor, src=0)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time

    datasize = case_config.ITERS * (Melements * 1024 * 1024 * 4 / 1E9)
    bandwidth = datasize / elapsed_time
    bandwidth_gib = bandwidth * 1E9 / (1024**3)
    
    return round(bandwidth, 2), round(bandwidth_gib, 2)

if __name__ == "__main__":
    config = parse_args()
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)
    with open(os.path.join(config.vendor, "case_config.yaml"), "r") as file:
        case_config_vendor = yaml.safe_load(file)
    case_config.update(case_config_vendor)
    case_config = Namespace(**case_config)
    select_gpus = [0, 8]
    dist.init_process_group(backend=case_config.DIST_BACKEND)  
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % config.node_size
    if local_rank == 0:
        if "mthreads" in config.vendor:
            torch.musa.set_device(local_rank)
        else:    
            torch.cuda.set_device(local_rank)
        print(rank, world_size, local_rank)

    gb, gib = main(config, case_config, rank, world_size, local_rank)
    multi_device_sync(config.vendor)
    if dist.get_rank() % config.node_size == 0:
        print(r"[FlagPerf Result]Rank {}'s inferconnect-P2P_intraserver-bandwidth=".format(dist.get_rank()) + str(gb) + "GB/s")
        print(r"[FlagPerf Result]Rank {}'s inferconnect-P2P_intraserver-bandwidth=".format(dist.get_rank()) + str(gib) + "GiB/s")

    dist.destroy_process_group()



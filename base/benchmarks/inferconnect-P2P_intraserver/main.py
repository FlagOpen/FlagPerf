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
    

def main(config, case_config, rank, world_size, local_rank, select_gpus):    
    set_ieee_float32(config.vendor)
    if rank == 0:
        print("finish initialization")
    
    Melements = case_config.Melements
    torchsize = (Melements, 1024, 1024)
    tensor = torch.rand(torchsize, dtype=torch.float32).to(local_rank)

    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")

    for _ in range(case_config.WARMUP):
        if local_rank == select_gpus[0]:
            dist.send(tensor, dst=select_gpus[1])
        elif local_rank == select_gpus[1]:
            dist.recv(tensor, src=select_gpus[0])
        
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    start_time = time.perf_counter()

    for _ in range(case_config.ITERS):
        if local_rank == select_gpus[0]:
            dist.send(tensor, dst=select_gpus[1])
        elif local_rank == select_gpus[1]:
            dist.recv(tensor, src=select_gpus[0])
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time

    datasize = case_config.ITERS * 2 * (Melements * 1024 * 1024 * 4 / 1E9)
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
    select_gpus = [0, 1]
    dist.init_process_group(backend=case_config.DIST_BACKEND)  
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % config.node_size      
    gb, gib = main(config, case_config, rank, world_size, local_rank, select_gpus)
    multi_device_sync(config.vendor)
    for output_rank in range(config.node_size):
        if local_rank in select_gpus and local_rank == output_rank:
            print(r"[FlagPerf Result]Rank {}'s inferconnect-P2P_intraserver-bandwidth=".format(dist.get_rank()) + str(gb) + "GB/s")
            print(r"[FlagPerf Result]Rank {}'s inferconnect-P2P_intraserver-bandwidth=".format(dist.get_rank()) + str(gib) + "GiB/s")
        multi_device_sync(config.vendor)
        
    dist.destroy_process_group()



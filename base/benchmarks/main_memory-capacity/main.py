# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

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
    device = torch.device('cuda:{}'.format(local_rank))
    byte_size = case_config.INITSIZE
    min_byte_size = 1
    total_allocated = 0
    allocated_tensors = []

    print(f"Init tensor size: {byte_size} MiB...")

    while byte_size >= min_byte_size:
        try:
            tensor = torch.empty(((byte_size * 1024 * 1024) // 4), dtype=torch.float32, device=device)
            allocated_tensors.append(tensor)
            total_allocated += byte_size
            print(f"Allocated: {total_allocated} MiB")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM at tensor size {byte_size} MiB. Allocated:{total_allocated} MiB")
                byte_size //= 2
                if byte_size < min_byte_size:
                    print("Tensor size == 1 Byte, finish test.")
                    break
                else:
                    print(f"Reduce tensor size to {byte_size} MiB")
            else:
                raise

    start = time.time()
    while time.time() <= start + 300:
        foo_str = "Waiting for power monitor"
    
    if local_rank == 0:
        print("Test Finished")
    
    return total_allocated


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
      
    mib = main(config, case_config, rank, world_size, local_rank)
    gib = round(mib / 1024, 2)
    gb = round((mib * 1048576) / 1000000000, 2)
    
    multi_device_sync(config.vendor)
    for output_rank in range(config.node_size):
        if local_rank == output_rank:
            print(r"[FlagPerf Result]Rank {}'s main_memory-capacity=".format(dist.get_rank()) + str(gb) + "GB")
            print(r"[FlagPerf Result]Rank {}'s main_memory-capacity=".format(dist.get_rank()) + str(gib) + "GiB")
        multi_device_sync(config.vendor)




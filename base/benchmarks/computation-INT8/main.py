import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.quantization
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

class QuantizedMatMulModel(nn.Module):
    def __init__(self, input_features, output_features):
        super(QuantizedMatMulModel, self).__init__()
        self.fc = nn.Linear(input_features, output_features)
    
    def forward(self, x):
        return self.fc(x)
    

def main(config, case_config, rank, world_size, local_rank):    
    if rank == 0:
        print("finish initialization")
    
    m = case_config.M
    n = case_config.N
    k = case_config.K
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    model = QuantizedMatMulModel(n, k).to(device)
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model, inplace = False)

    x = torch.randn(m, n).to(device)
    model_prepared(x)

    model_quantized = torch.quantization.convert(model_prepared, inplace = False)

    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start warmup")
    
    for _ in range(case_config.WARMUP):
        _result = model_quantized(x)
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    if rank == 0:
        print("start test")
    
    host_device_sync(config.vendor)
    multi_device_sync(config.vendor)
    start_time = time.perf_counter()
    
    for _ in range(case_config.ITERS):
        _result = model_quantized(x)
    
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
        multi_device_sync(config.vendor)
        
    dist.destroy_process_group()


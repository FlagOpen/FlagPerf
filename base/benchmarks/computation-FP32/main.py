import torch
import torch.distributed as dist
import os
import time
from argparse import ArgumentParser, Namespace
import yaml


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args
    

def main(config, case_config):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    time.sleep(100)

    return 19.1


if __name__ == "__main__":    
    config = parse_args()
    with open("case_config.yaml", "r") as file:
        case_config = yaml.safe_load(file)
    with open(os.path.join(config.vendor, "case_config.yaml"), "r") as file:
        case_config_vendor = yaml.safe_load(file)
    case_config.update(case_config_vendor)
    case_config = Namespace(**case_config)
        
    dist.init_process_group(backend=case_config.DIST_BACKEND)    
    result = main(config, case_config)
    print(r"[FlagPerf Result]computation-FP32=" + str(result) + "TFLOPS")
    dist.destroy_process_group()



import os
import subprocess
from argparse import ArgumentParser
import importlib
from typing import Mapping
import inspect
from loguru import logger
from collections import namedtuple
import time


def parse_args():

    parser = ArgumentParser(description="aquila_in_container")
    parser.add_argument("--vendor_config", type=str, required=True)
    parser.add_argument("--hosts", nargs="+", type=str, required=True)
    parser.add_argument("--master_port", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def is_property(name: str, value):
    return all([
        not name.startswith('__'),
        not callable(value),
        not inspect.isclass(value),
        not inspect.ismodule(value),
        not inspect.ismethod(value),
        not inspect.isfunction(value),
        not inspect.isbuiltin(value),
    ])


def get_properties_from_config(config):
    if not isinstance(config, Mapping):
        config = config.__dict__
    properties = dict()
    for name, value in config.items():
        if is_property(name, value):
            properties[name] = value

    return properties


def merge_config(args):
    base_config = importlib.import_module("config")
    base_config_dict = {}
    for name, value in get_properties_from_config(base_config).items():
        base_config_dict[name] = value
    
    spec = importlib.util.spec_from_file_location("config", args.vendor_config)
    vendor_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor_config)
    for name, value in get_properties_from_config(vendor_config).items():
        if name not in base_config_dict.keys():
            logger.warning("new config item: " + name)
            base_config_dict[name] = value
        else:
            logger.warning("update item " + name + " from " +
                           str(base_config_dict[name]) + "to" + str(value))
            base_config_dict[name] = value
    return base_config_dict


if __name__ == "__main__":
    args = parse_args()
    logger.debug(args)
    config = merge_config(args)
    logger.debug(config)
    
    Config = namedtuple("Config", config.keys())
    config = Config(**config)
    
    flagscale_home = os.path.join(config.DATA_DIR, config.FLAGSCALE_HOME)
    
    start_time = time.time()
    noderank = 0
    for ip in args.hosts:

        exec_cmd = "bash singlenode_run.sh"
        exec_cmd = exec_cmd + " " + flagscale_home
        exec_cmd = exec_cmd + " " + config.DATA_DIR
        exec_cmd = exec_cmd + " " + config.DATASET
        exec_cmd = exec_cmd + " " + str(int(config.TRAINING_TOKENS) // int(config.SEQLENGTH))
        exec_cmd = exec_cmd + " " + str(config.TENSOR_PARALLEL)
        exec_cmd = exec_cmd + " " + str(config.PIPELINE_PARALLEL)
        exec_cmd = exec_cmd + " " + str(config.MICRO_BATCHSIZE)
        exec_cmd = exec_cmd + " " + str(config.GLOBAL_BATCHSIZE)
        exec_cmd = exec_cmd + " " + str(noderank)
        exec_cmd = exec_cmd + " " + str(len(args.hosts))
        exec_cmd = exec_cmd + " " + args.hosts[0]
        exec_cmd = exec_cmd + " " + args.master_port
        exec_cmd = exec_cmd + " " + config.CUDA_DEVICE_MAX_CONNECTIONS
        exec_cmd = exec_cmd + " " + config.NCCL_SOCKET_IFNAME
        exec_cmd = exec_cmd + " " + config.NCCL_IB_DISABLE
        exec_cmd = exec_cmd + " " + config.NCCL_IB_CUDA_SUPPORT
        exec_cmd = exec_cmd + " " + config.NCCL_IB_GID_INDEX
        exec_cmd = exec_cmd + " " + config.NCCL_IB_HCA
        exec_cmd = exec_cmd + " " + config.NCCL_DEBUG
        
        logger.info(ip)
        logger.info(exec_cmd)
        logger.info("")
        
        with open(os.path.join(args.log_dir, "noderank" + str(noderank) + ".log.txt"), "w") as f:
            p = subprocess.Popen(exec_cmd,
                                 shell=True,
                                 stdout=f,
                                 stderr=subprocess.STDOUT)
            p.wait()
        
        noderank += 1
    training_time = time.time() - start_time
    system_throughput = float(config.TRAINING_TOKENS) / training_time
    world_size = len(args.hosts) * 8
    chip_throughput = system_throughput / world_size
    MFU = chip_throughput * 6 * 7000000000 / float(config.flops_16bit)
    tflops_throughput = chip_throughput / float(config.flops_16bit) * 1e12
    logger.info("Throughput(token per chip per second): " + str(chip_throughput))
    logger.info("MFU: " + str(MFU))
    logger.info("Throughput(token per TFLOPS): " + str(tflops_throughput))

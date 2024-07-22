# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import subprocess
from argparse import ArgumentParser
import os
import sys
from importlib import import_module


def parse_args():
    '''we parse ddp related args, check system config args, and running env
       args such as --data_dir_xxx. Then pass all useful args to the real
       training script.
    '''
    parser = ArgumentParser(description="flagscale main python")
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--node_rank", type=int, required=True)
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=int, required=True)
    parser.add_argument("--vendor", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--flagperf_config_file", type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    sys.path.append(os.path.dirname(args.flagperf_config_file))
    config_file = os.path.basename(args.flagperf_config_file).split('.')[0]
    config_dir_path = os.path.dirname(args.flagperf_config_file)

    module = import_module(config_file)

    localbs = getattr(module, 'localbs')
    train_steps = getattr(module, 'train_steps')
    theoryflops = getattr(module, 'theoryflops')
    megapath = getattr(module, 'megatron_path')
    tensor_parallel = getattr(module, 'tensor_parallel')
    pipeline_parallel = getattr(module, 'pipeline_parallel')
    tokenizer = getattr(module, 'tokenizer_path')
    
    nnodes = args.nnodes
    nproc_per_node = args.nproc_per_node
    node_rank = args.node_rank
    master_addr = args.master_addr
    master_port = args.master_port
    data_dir = args.data_dir
    tokenizer_dir = os.path.join(args.data_dir, tokenizer)


    task_log_file = os.path.join(args.log_dir, "megatron.log.txt")
    
    # merge llama3 patch

    if args.vendor=="cambricon" or args.vendor=="metax":
        exec_cmd = "bash pretrain_llama3.sh"
    else:    
        origin_file = os.path.join(megapath, "megatron/training/arguments.py")
        exec_cmd = "patch --silent --forward " + origin_file + " arguments.patch -o tmp.py;mv tmp.py " + origin_file
        exec_cmd = exec_cmd + ";"
        
        origin_file = os.path.join(megapath, "megatron/training/tokenizer/tokenizer.py")
        exec_cmd = exec_cmd + "patch --silent --forward " + origin_file + " tokenizer.patch -o tmp.py;mv tmp.py " + origin_file
        exec_cmd = exec_cmd + ";"
        
        # bash pretrain_llama3.sh
        
        exec_cmd = exec_cmd + "bash pretrain_llama3.sh"
    
    # args

    exec_cmd = exec_cmd + " " + data_dir
    exec_cmd = exec_cmd + " " + str(nproc_per_node)
    exec_cmd = exec_cmd + " " + str(nnodes)
    exec_cmd = exec_cmd + " " + str(node_rank)
    exec_cmd = exec_cmd + " " + str(master_addr)
    exec_cmd = exec_cmd + " " + str(master_port)
    exec_cmd = exec_cmd + " " + megapath
    exec_cmd = exec_cmd + " " + str(localbs)
    exec_cmd = exec_cmd + " " + str(train_steps)
    exec_cmd = exec_cmd + " " + str(tensor_parallel)
    exec_cmd = exec_cmd + " " + str(pipeline_parallel)
    exec_cmd = exec_cmd + " " + str(tokenizer_dir)
    exec_cmd = exec_cmd + " " + os.path.join(config_dir_path, "training_adapter.sh")
    print(exec_cmd)

    with open(task_log_file, "w") as f:
        p = subprocess.Popen(exec_cmd,
                             shell=True,
                             stdout=f,
                             stderr=subprocess.STDOUT)
        p.wait()

    time_per_step = -1.0
    with open(task_log_file) as f:
        for line in f.readlines():
            if "elapsed time per iteration (ms): " in line:
                info = line.split("|")[2]
                steptime = info.split(":")[1]
                time_per_step = float(steptime) / 1000

    whole_tps = 512 * 8192 / time_per_step
    chip_tps = whole_tps / (args.nproc_per_node * args.nnodes)
    print("System tokens per second: ", whole_tps)
    print("Tokens/p/s: ", chip_tps)
    print("MFU: ", chip_tps * 8000000000.0 * 6 / theoryflops)

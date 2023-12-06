#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''This script is called in container to execute the real training task.
   Support pytorch DDP only.
'''
import os
import sys
import subprocess
from argparse import ArgumentParser

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from utils import flagperf_logger
from utils import start_task_helper as helper

START_LOGGER = flagperf_logger.FlagPerfLogger()


def parse_args():
    '''we parse ddp related args, check system config args, and running env
       args such as --data_dir_xxx. Then pass all useful args to the real
       training script.
    '''
    parser = ArgumentParser(description="Start pytorch training process. ")
    parser.add_argument("--node_rank",
                        type=int,
                        default=0,
                        help="The rank of the node for multi-node distributed "
                        "training")
    parser.add_argument("--master_addr",
                        default="127.0.0.1",
                        type=str,
                        help="Master node (rank 0)'s address, should be either"
                        "the IP address or the hostname of node 0, for "
                        "single node multi-proc training, the "
                        "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port",
                        default=29501,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communication during distributed "
                        "training")
    parser.add_argument("--nnodes",
                        type=int,
                        required=True,
                        help="how many hosts to run the testcase.")
    parser.add_argument("--nproc",
                        type=int,
                        required=True,
                        help="how many processes will run on each host.")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="The accelerator vendor that run the located.")
    parser.add_argument("--visible_dev_env",
                        type=str,
                        default=None,
                        help="The accelerator XXX_VISIBLE_DEVICE env name.")
    parser.add_argument("--case_name",
                        type=str,
                        required=True,
                        help="Name of testcase.")
    parser.add_argument("--round",
                        type=int,
                        required=True,
                        help="round of testcase, for repeating test.")
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="The model name of testcase.")
    parser.add_argument("--host_addr",
                        type=str,
                        required=True,
                        help="The host address that start task.")
    parser.add_argument("--train_script",
                        type=str,
                        required=True,
                        help="The training script to start by this launcher.")
    parser.add_argument("--enable_extern_config",
                        action="store_true",
                        help="Sets to enable non-standard config parameters.")
    parser.add_argument("--extern_config_file",
                        type=str,
                        required=True,
                        help="The testcase config file.")
    parser.add_argument("--data_dir",
                        type=str,
                        default="/mnt/dataset/",
                        help="Data directory.")
    parser.add_argument("--log_dir",
                        type=str,
                        default="/workspace/flagperf/training/result/",
                        help="Log directory in container.")
    parser.add_argument("--log_level",
                        type=str,
                        default="debug",
                        help="Log level.")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def main():
    '''Parse args and start the training task. Support DDP.
    '''
    task_args = parse_args()
    task_args.framework = "deepspeed"

    task_log_dir = helper.init_flagperf_logger(START_LOGGER, task_args)
    helper.write_pid_file(task_args.log_dir, "start_deepspeed_task.pid")

    train_script_path = helper.get_train_script_path(task_args)
    config_dir, config_file = helper.get_config_dir_file(task_args)
    config_file = os.path.join(config_dir, config_file)
    ds_config_file = os.path.join(config_dir, "ds_config.json")

    exec_cmd = "cd " + os.path.dirname(train_script_path) + ";"
    exec_cmd = exec_cmd + "deepspeed --num_gpus=" + str(
        task_args.nproc) + " run_pretraining.py"

    exec_cmd = exec_cmd + " --deepspeed --deepspeed_config "
    exec_cmd = exec_cmd + ds_config_file
    exec_cmd = exec_cmd + " --data_dir " + task_args.data_dir

    exec_cmd = exec_cmd + " --flagperf_config " + config_file
    exec_cmd = exec_cmd + " --nproc " + str(
        task_args.nproc) + " --nnodes " + str(task_args.nnodes)

    task_log_file = os.path.join(task_log_dir, "rank0.log.txt")

    with open(task_log_file, "w") as f:
        p = subprocess.Popen(exec_cmd,
                             shell=True,
                             stdout=f,
                             stderr=subprocess.STDOUT)
        p.wait()


if __name__ == '__main__':
    main()

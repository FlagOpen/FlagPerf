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


def _set_common_ddp_envs(task_args):
    '''Set and return common ddp env items
    '''
    # world size in terms of number of processes
    dist_world_size = task_args.nproc * task_args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = str(task_args.master_addr)
    current_env["MASTER_PORT"] = str(task_args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["NODE_RANK"] = str(task_args.node_rank)

    # set GPU/MLU device env, TODO other vendor's device
    if task_args.visible_dev_env is not None:
        acce_visible = range(0, task_args.nproc)
        current_env[task_args.visible_dev_env] = ",".join(
            str(_id) for _id in acce_visible)
    return current_env


def _get_basic_train_script_args(task_args):
    '''Generate basic train script args according to the script options.'''
    config_dir, config_file = helper.get_config_dir_file(task_args)
    if config_dir is None or config_file is None:
        START_LOGGER.error(
            f"Can't find config dir :{config_dir} or config file:{config_file}."
        )
        return None
    if task_args.enable_extern_config:
        extern_module_dir = helper.get_extern_module_dir(task_args)
        if extern_module_dir is None:
            START_LOGGER.error("Can't find extern module dir.")
            return None

    basic_train_script_args = " --extern_config_dir " + config_dir \
                              + " --extern_config_file " + config_file \
                              + " --vendor " + task_args.vendor \
                              + " --data_dir " + task_args.data_dir
    if task_args.enable_extern_config:
        basic_train_script_args += " --enable_extern_config " \
                                   + "--extern_module_dir " + extern_module_dir
    return basic_train_script_args


def main():
    '''Parse args and start the training task. Support DDP.
    '''
    task_args = parse_args()
    task_args.framework = "pytorch"

    task_log_dir = helper.init_flagperf_logger(START_LOGGER, task_args)
    helper.write_pid_file(task_args.log_dir, "start_pytorch_task.pid")

    # Check and get train script & its basic args.
    basic_train_script_args = _get_basic_train_script_args(task_args)
    if basic_train_script_args is None:
        START_LOGGER.error("Can't get args of train script.")
        sys.exit(3)

    train_script_path = helper.get_train_script_path(task_args)
    if train_script_path is None:
        START_LOGGER.error("Can't find path of train script.")
        sys.exit(4)

    current_env = _set_common_ddp_envs(task_args)

    # start all processes in container
    processes = []
    for local_rank in range(0, task_args.nproc):
        dist_rank = task_args.nproc * task_args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        start_cmd = sys.executable + " -u " + train_script_path + " " \
                                   + basic_train_script_args + " 2>&1 | tee " \
                                   + task_log_dir + "/rank" + str(dist_rank) \
                                   + ".out.log"

        # Start all the processes, TODO debug multi proc, stdout, stderr
        START_LOGGER.info("Start task with command: " + start_cmd)
        START_LOGGER.debug("----------- Process envs -----------")
        for environ in current_env.keys():
            START_LOGGER.debug(environ + ":" + current_env[environ])
        START_LOGGER.debug("start command: " + start_cmd)
        process = subprocess.Popen(start_cmd, shell=True, env=current_env)
        processes.append(process)

    for proc in processes:
        proc.wait()

    START_LOGGER.stop()
    # check the return code of each process
    if proc.returncode != 0:
        raise Exception("Exception! process {0} returncode != 0, please check relevant issue".format(str(proc.pid)))


if __name__ == '__main__':
    main()

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

START_LOGGER = flagperf_logger.FlagPerfLogger()


def _get_model_path(vendor, test_type, model_name):
    '''Return the model path according to vendor or None if it doesn't exist.
    '''
    if test_type == "default":
        model_path = os.path.join(CURR_PATH + "/../../benchmarks/" + model_name
                                  + "/pytorch/")
    else:
        model_path = os.path.join(CURR_PATH + "/../../" + vendor + "/benchmarks/"
                                  + model_name + "/pytorch/")
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        START_LOGGER.error("Can't find model path: " + model_path)
        return None
    return model_path


def _get_config_dir_file(task_args):
    '''Return config path and file path in vendor's dir, or None if the config
       file does not exist.
    '''
    config_file = task_args.extern_config_file
    config_dir = os.path.join(CURR_PATH + "/../../" + task_args.vendor
                              + "/", task_args.model_name
                              + "-pytorch/config/")
    config_dir = os.path.abspath(config_dir)
    if not os.path.isfile(os.path.join(config_dir, config_file)):
        START_LOGGER.error("Can't find config file: " + config_file + " in "
                           + config_dir)
        return None, None
    return config_dir, config_file


def _get_train_script_path(task_args):
    '''Return training script path, or None if it does not exist.'''
    model_path = _get_model_path(task_args.vendor, task_args.test_type,
                                 task_args.model_name)
    if model_path is None:
        return None
    train_script_path = os.path.join(model_path, task_args.train_script)
    if not os.path.isfile(train_script_path):
        START_LOGGER.error("Can't find training strcipt:" + train_script_path)
        return None
    return train_script_path


def _get_extern_module_dir(task_args):
    '''Return extern module dir or None if something wrong.'''
    extern_module_dir = os.path.join(CURR_PATH + "/../../" + task_args.vendor,
                                     task_args.model_name + "-pytorch/extern/")
    extern_module_dir = os.path.abspath(extern_module_dir)
    if not os.path.isdir(extern_module_dir):
        START_LOGGER.error("Can't find extern module dir:" + extern_module_dir)
        return None
    return extern_module_dir


def parse_args():
    '''we parse ddp related args, check system config args, and running env
       args such as --data_dir_xxx. Then pass all useful args to the real
       training script.
    '''
    parser = ArgumentParser(description="Start pytorch training porcess. ")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either"
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29501, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    parser.add_argument("--nnodes", type=int, required=True,
                        help="how many hosts to run the testcase.")
    parser.add_argument("--nproc", type=int, required=True,
                        help="how many processes will run on each host.")

    parser.add_argument("--vendor", type=str, required=True,
                        help="The accelerator vendor that run the located.")
    parser.add_argument("--visible_dev_env", type=str, default=None,
                        help="The accelerator XXX_VISIBLE_DEVICE env name.")
    parser.add_argument("--test_type", type=str, default="default",
                        help="Test type of the benchmark. It should be "
                             "\"default\" or \"customized\"")
    parser.add_argument("--case_name", type=str, required=True,
                        help="Name of testcase.")
    parser.add_argument("--round", type=int, required=True,
                        help="round of testcase, for repeating test.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="The model name of testcase.")
    parser.add_argument("--host_addr", type=str, required=True,
                        help="The host address that start task.")
    parser.add_argument("--train_script", type=str, required=True,
                        help="The training script to start by this launcher.")
    parser.add_argument("--enable_extern_config", action="store_true",
                        help="Sets to enable non-standard config parameters.")
    parser.add_argument("--extern_config_file", type=str, required=True,
                        help="The testcase config file.")
    parser.add_argument("--data_dir", type=str, default="/mnt/dataset/",
                        help="Data directory.")
    parser.add_argument("--log_dir", type=str,
                        default="/workspace/flagperf/training/result/",
                        help="Log directory in container.")
    parser.add_argument("--log_level", type=str, default="debug",
                        help="Log level.")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def write_pid_file(pid_file_path, pid_file):
    '''Write pid file for watching the process later.
       In each round of case, we will write the current pid in the same path.
    '''
    pid_file_path = os.path.join(pid_file_path, pid_file)
    if os.path.exists(pid_file_path):
        os.remove(pid_file_path)
    file_d = open(pid_file_path, "w")
    file_d.write("%s\n" % os.getpid())
    file_d.close()


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
        current_env[task_args.visible_dev_env] = ",".join(str(_id) for _id in
                                                          acce_visible)
    return current_env


def _get_basic_train_script_args(task_args):
    '''Generate basic train script args according to the script options.'''
    config_dir, config_file = _get_config_dir_file(task_args)
    if config_dir is None or config_file is None:
        return None
    if task_args.enable_extern_config:
        extern_module_dir = _get_extern_module_dir(task_args)
        if extern_module_dir is None:
            return None

    basic_train_script_args = " --data_dir " + task_args.data_dir \
                              + " --extern_config_dir " + config_dir \
                              + " --extern_config_file " + config_file

    if task_args.enable_extern_config:
        basic_train_script_args += " --enable_extern_config " \
                                   + "--extern_module_dir " + extern_module_dir
    return basic_train_script_args


def main():
    '''Parse args and start the training task. Support DDP.
    '''
    task_args = parse_args()

    # Create logger. We don't need to check the log path because logger.init()
    # will make it.
    task_log_dir = os.path.join(task_args.log_dir, task_args.case_name + "/"
                                + "round" + str(task_args.round) + "/"
                                + task_args.host_addr + "_noderank"
                                + str(task_args.node_rank))
    START_LOGGER.init(task_log_dir, "start_pytorch_task.log",
                      task_args.log_level, "both", log_caller=True)
    START_LOGGER.info(",".join(task_args.__dict__))
    write_pid_file(task_args.log_dir, "start_pytorch_task.pid")

    # Check and get train script & its basic args.
    basic_train_script_args = _get_basic_train_script_args(task_args)
    train_script_path = _get_train_script_path(task_args)
    if train_script_path is None or basic_train_script_args is None:
        sys.exit(3)

    current_env = _set_common_ddp_envs(task_args)

    # start all processes in container
    processes = []
    for local_rank in range(0, task_args.nproc):
        dist_rank = task_args.nproc * task_args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        start_cmd = sys.executable + " -u " + train_script_path + " " \
                                   + basic_train_script_args + " |& tee " \
                                   + task_log_dir + "/rank" + str(dist_rank) \
                                   + ".out.log"

        # Start all the processes, TODO debug multi proc, stdout, stderr
        START_LOGGER.info("Start task with command: " + start_cmd)
        START_LOGGER.debug("----------- Process envs -----------")
        for environ in current_env.keys():
            START_LOGGER.debug(environ + ":" + current_env[environ])
        process = subprocess.Popen(start_cmd, shell=True, env=current_env)
        processes.append(process)

    for proc in processes:
        proc.wait()

    START_LOGGER.stop()


if __name__ == '__main__':
    main()

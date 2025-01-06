# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from loguru import logger
import os
import sys
from argparse import ArgumentParser
import subprocess


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--case_name",
                        type=str,
                        required=True,
                        help="case name")

    parser.add_argument("--nnodes",
                        type=int,
                        required=True,
                        help="number of node")

    parser.add_argument("--nproc_per_node",
                        type=int,
                        required=True,
                        help="*pu per node")

    parser.add_argument("--log_dir",
                        type=str,
                        required=True,
                        help="abs log dir")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    parser.add_argument("--mode",
                        type=str,
                        required=True,
                        help="mode like cpu")

    parser.add_argument("--warmup",
                        type=str,
                        required=True,
                        help="warmup")

    parser.add_argument("--log_level",
                        type=str,
                        required=True,
                        help="log level")

    parser.add_argument("--master_port",
                        type=int,
                        required=True,
                        help="master port")

    parser.add_argument("--master_addr",
                        type=str,
                        required=True,
                        help="master ip")

    parser.add_argument("--host_addr", type=str, required=True, help="my ip")

    parser.add_argument("--node_rank", type=int, required=True, help="my rank")

    parser.add_argument("--perf_path",
                        type=str,
                        required=True,
                        help="abs path for FlagPerf/base")

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


if __name__ == "__main__":
    config = parse_args()

    logfile = os.path.join(
        config.log_dir, config.case_name,
        config.host_addr + "_noderank" + str(config.node_rank),
        "container_main.log.txt")
    logger.remove()
    logger.add(logfile, level=config.log_level)
    logger.add(sys.stdout, level=config.log_level)

    logger.info(config)
    write_pid_file(config.log_dir, "start_base_task.pid")
    logger.info("Success Writing PID file at " +
                os.path.join(config.log_dir, "start_base_task.pid"))

    test_file, op, dataformat, spectflops, oplib, chip = config.case_name.split(":")

    case_dir = os.path.join(config.perf_path, "benchmarks", test_file)
    start_cmd = "cd " + case_dir + ";python3 main.py "
    start_cmd += " --vendor=" + config.vendor
    start_cmd += " --case_name=" + op
    start_cmd += " --spectflops=" + spectflops
    start_cmd += " --dataformat=" + dataformat
    start_cmd += " --oplib=" + oplib
    start_cmd += " --chip=" + chip
    # start_cmd += " --mode=" + config.mode
    # start_cmd += " --warmup=" + config.warmup

    script_log_file = os.path.join(os.path.dirname(logfile),
                                   "operation.log.txt")
    logger.info("print operation.log.txt file")
    logger.info(script_log_file)
    logger.info(start_cmd)
    logger.info(script_log_file)

    f = open(script_log_file, "w")
    p = subprocess.Popen(start_cmd,
                         shell=True,
                         stdout=f,
                         stderr=subprocess.STDOUT)
    # 获取日志
    flaggems_dir = os.getenv("FLAGGEMS_WORK_DIR", "/")
    logger.info("FLAGGEMS_WORK_DIR======flaggems_dir")
    logger.info(flaggems_dir)
    gems_repo = subprocess.check_output(
        ["find", flaggems_dir, "-type", "d", "-name", "FlagGems"], text=True).strip()
    logger.info("gems_repo=========")
    logger.info(gems_repo)
    # log_dir = os.path.join(gems_repo, "benchmark", "result--level_core--record_log")
    log_dir = os.path.join(gems_repo, "benchmark", "result_test_blas_perf--level_core--mode_cpu--warmup_0--record_log.log")
    # log_dir = os.path.join(gems_repo, "benchmark", "result-m_mm--level_core--mode_cpu--warmup_1000--record_log-s.log")
    logger.info("log_dir=========")
    logger.info(log_dir)
    save_path = os.path.join(os.path.dirname(logfile),
                                   "result.log.txt")
    with open(log_dir, "r", encoding="utf-8") as file_r, open(save_path, "w", encoding="utf-8") as file_w:
        for line in file_r:
            logger.info("result.log.txt print ok")
            file_w.write(line + '\n')

    p.wait()
    f.close()
    logger.info("Task Finish")

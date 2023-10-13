# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib
from loguru import logger
import time
import os
import sys
from argparse import ArgumentParser
import subprocess


def init_logger(config):
    logger.remove()
    logger.level("Finish Info", no=50)

    logdir = config.log_dir
    logfile = logdir + "/container.out.log"
    logger.add(logfile, level=config.loglevel)

    logger.add(sys.stdout, level=config.loglevel)


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--perf_dir",
                        type=str,
                        required=True,
                        help="abs dir of FlagPerf/inference/")

    parser.add_argument("--data_dir",
                        type=str,
                        required=True,
                        help="abs dir of data used in dataloader")

    parser.add_argument("--log_dir",
                        type=str,
                        required=True,
                        help="abs dir to write log")

    parser.add_argument("--loglevel",
                        type=str,
                        required=True,
                        help="DEBUG/INFO/WARNING/ERROR")

    parser.add_argument("--case",
                        type=str,
                        required=True,
                        help="case name like resnet50")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    parser.add_argument("--framework",
                        type=str,
                        required=True,
                        help="validation framework name like pytorch")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


if __name__ == "__main__":
    config_from_args = parse_args()
    config_from_args.framework = config_from_args.framework.split('_')[0]
    
    init_logger(config_from_args)
    
    e2e_start = time.time()
    
    logger.info(config_from_args)
    command = "cd " + config_from_args.perf_dir
    command = command + ";cd benchmarks/" + config_from_args.case
    command = command + "/flagscale;bash main.sh " + config_from_args.data_dir
    logger.info(command)
    p = subprocess.Popen(command, shell=True)
    p.wait()
    
    e2e_time = time.time() - e2e_start
    e2e_time = round(float(e2e_time), 3)
    
    flagscale_logfile = config_from_args.log_dir + "/stdout_err.out.log"
    loss = -1.0
    wholeperf = -1.0
    coreperf = -1.0
    
    logfile = open(flagscale_logfile)
    for line in logfile.readlines():
        if 'avg loss' in line:
            loss = float(line.replace('\n', '').split(':')[1])
        if 'Core throughput' in line:
            coreperf = float(line.replace('\n', '').split(':')[1])
        if 'Whole throughput' in line:
            wholeperf = float(line.replace('\n', '').split(':')[1])
    logfile.close()
            
    infer_info = {
        "vendor": config_from_args.vendor,
        "precision": "16bit",
        "e2e_time(second)": e2e_time,
        "p_validation_whole(qps)": None,
        "p_validation_core(qps)": None,
        "p_inference_whole(qps)": wholeperf,
        "*p_inference_core(qps)": coreperf,
        "val_average_acc": None,
        "infer_average_loss": loss
    }
    logger.log("Finish Info", infer_info)
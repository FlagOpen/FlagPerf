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
                        
    parser.add_argument("--host_addr",
                        type=str,
                        required=True,
                        help="my ip")
    
    parser.add_argument("--node_rank",
                        type=int,
                        required=True,
                        help="my rank")
                        
    parser.add_argument("--bench_or_tool",
                        type=str,
                        required=True,
                        help="benchmarks or toolkits")
                        
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
    
    logfile = os.path.join(config.log_dir, config.case_name, config.host_addr + "_noderank" + str(config.node_rank), "container_main.log.txt")
    logger.remove()
    logger.add(logfile, level=config.log_level)
    logger.add(sys.stdout, level=config.log_level)
    
    logger.info(config)
    write_pid_file(config.log_dir, "start_base_task.pid")
    logger.info("Success Writing PID file at " + os.path.join(config.log_dir, "start_base_task.pid"))
    
    if config.bench_or_tool == "BENCHMARK":
        logger.info("Using PyTorch to Test {}'s {}".format(config.vendor, config.case_name))
        case_dir = os.path.join(config.perf_path, "benchmarks", config.case_name)
        
        start_cmd = "cd " + case_dir + ";torchrun"
        
        # for torch
        start_cmd += " --nproc_per_node=" + str(config.nproc_per_node)
        start_cmd += " --nnodes=" + str(config.nnodes)
        start_cmd += " --node_rank=" + str(config.node_rank)
        start_cmd += " --master_addr=" + str(config.master_addr)
        start_cmd += " --master_port=" + str(config.master_port)   
        
        # for flagperf
        start_cmd += " main.py"
        start_cmd += " --vendor=" + config.vendor
        start_cmd += " --node_size=" + str(config.nproc_per_node)
        
        script_log_file = os.path.join(os.path.dirname(logfile), "benchmark.log.txt")  
    elif config.bench_or_tool == "TOOLKIT":
        logger.info("Using {}'s Toolkits to Test {}".format(config.vendor, config.case_name))
        case_dir = os.path.join(config.perf_path, "toolkits", config.case_name, config.vendor)
        
        start_cmd = "cd " + case_dir + ";bash main.sh"
        
        script_log_file = os.path.join(os.path.dirname(logfile), "toolkit.log.txt")    
    else:
        logger.error("Invalid BENCHMARKS_OR_TOOLKITS CONFIG, STOPPED TEST!")
        exit(1)
    
    logger.info(start_cmd)
    logger.info(script_log_file)
    
    f = open(script_log_file, "w")
    p = subprocess.Popen(start_cmd,
                         shell=True,
                         stdout=f,
                         stderr=subprocess.STDOUT)
    p.wait()
    f.close() 
    logger.info("Task Finish")    
  

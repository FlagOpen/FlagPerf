import os
import sys
import subprocess
from config import *
from datetime import datetime
from loguru import logger


def usage():
    ''' Show usage and exit with exit_code. '''
    print("Usage: python3 ", __file__)
    print("Edit config file test_conf.py & cluster_conf.py in "
          "training/run_benchmarks/config and run.")
    sys.exit(0)


def print_welcome_msg():
    '''Print colorful welcome message to console.'''
    print("\033[1;34;40m==============================================\033[0m")
    print("\033[1;36;40m          Welcome to FlagPerf!\033[0m")
    print(
        "\033[1;36;40m      See more at https://github.com/FlagOpen/FlagPerf \033[0m"
    )
    print("\033[1;34;40m==============================================\033[0m")


def main():
    print_welcome_msg()

    entrance_file = os.path.join(flagperf_home, "training/benchmarks/",
                                 casename, "in_container/run_pretraining.py")
    case_framework = casename + "-" + "in_container"
    config_file = os.path.join(flagperf_home, "training", vendor,
                               case_framework, "config.py")

    entrance_path = os.path.dirname(entrance_file)

    logdir = os.path.join(flagperf_home, "training", log_dir)
    timestamp_str = (datetime.now()).strftime("%Y%m%d%H%M%S")
    log_dir_stamp = os.path.join(logdir, "run" + timestamp_str)
    os.makedirs(log_dir_stamp, exist_ok=True)

    exec_cmd = "cd " + entrance_path + ";"
    exec_cmd = exec_cmd + "python3 run_pretraining.py"
    exec_cmd = exec_cmd + " --vendor_config " + config_file
    exec_cmd = exec_cmd + " --hosts "
    for ip in hosts:
        exec_cmd = exec_cmd + ip + " "
    exec_cmd = exec_cmd + " --master_port " + str(master_port)
    exec_cmd = exec_cmd + " --perf_dir " + flagperf_home
    exec_cmd = exec_cmd + " --log_dir " + log_dir_stamp

    logger.debug(exec_cmd)
    
    logger.info("Task has been started, waiting...")
    with open(os.path.join(log_dir_stamp, "flagperf.log.txt"), "w") as f:
        p = subprocess.Popen(exec_cmd,
                             shell=True,
                             stdout=f,
                             stderr=subprocess.STDOUT)
        p.wait()
    logger.info("Test Finished")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        usage()
    main()

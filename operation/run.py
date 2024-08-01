# Copyright (c) 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
''' TODO Copyright and Other info '''

import os
import sys
import time
import getpass
import yaml
from argparse import Namespace
import importlib
import json
import numpy as np

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../")))
from utils import cluster_manager
from utils import flagperf_logger
from utils import image_manager

VERSION = "v0.1"
RUN_LOGGER = flagperf_logger.FlagPerfLogger()
CLUSTER_MGR = cluster_manager.ClusterManager()


def usage():
    ''' Show usage and exit with exit_code. '''
    print("Usage: python3 ", __file__)
    print("Edit config file config/host.yaml in and run.")
    sys.exit(0)


def check_cluster_health():
    ''' Try to ssh login to all the hosts in cluster_conf.hosts.
        Return None if everything goes well.
    '''
    RUN_LOGGER.debug("Cluster healthcheck ssh. Hosts are: " +
                     ",".join(CLUSTER_MGR.get_hosts_list()))
    bad_hosts = CLUSTER_MGR.healthcheck()
    if len(bad_hosts) != 0:
        for bad_host in bad_hosts:
            RUN_LOGGER.error("Check " + bad_host + " failed. ssh command exit "
                             "with: " + str(bad_hosts[bad_host]))
        RUN_LOGGER.error("Check hosts in the cluster......[FAILED] [EXIT]")
        sys.exit(3)
    RUN_LOGGER.info("Check hosts in the cluster......[SUCCESS]")


def check_cluster_deploy_path(dp_path):
    '''Make sure that flagperf is deployed on all the hosts
    '''
    RUN_LOGGER.debug("Check flagperf deployment path: " + dp_path)
    bad_hosts = CLUSTER_MGR.run_command_all_hosts("cd " + dp_path)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't find deployed path: " +
                         ",".join(bad_hosts.keys()))
        RUN_LOGGER.error("Check cluster deploy path " + dp_path +
                         "......[FAILED] [EXIT]")
        sys.exit(3)
    RUN_LOGGER.info("Check flagperf deployment path: " + dp_path +
                    "...[SUCCESS]")


def prepare_docker_image_cluster(dp_path, image_mgr, framework, nnodes,
                                 config):
    '''Prepare docker image in registry and in the cluster.
    '''
    vendor = config.VENDOR
    image_vendor_dir = os.path.join(dp_path, "vendors", vendor, framework)
    image_name = image_mgr.repository + ":" + image_mgr.tag
    RUN_LOGGER.debug("Prepare docker image in cluster. image_name=" +
                     image_name + " image_vendor_dir=" + image_vendor_dir)
    prepare_image_cmd = "cd " + dp_path + " && " + sys.executable \
                        + " ../utils/image_manager.py -o build -i " \
                        + image_mgr.repository + " -t " + image_mgr.tag \
                        + " -d " + image_vendor_dir + " -f " + framework
    timeout = 1200
    RUN_LOGGER.debug("Run cmd in the cluster to prepare docker image: " +
                     prepare_image_cmd + " timeout=" + str(timeout))
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(prepare_image_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't pull image: " +
                         ",".join(bad_hosts.keys()))
        return False
    return True


def start_container_in_cluster(dp_path, run_args, container_name, image_name,
                               nnodes):
    '''Call CLUSTER_MGR tool to start containers.'''
    start_cmd = "cd " + dp_path + " && " + sys.executable \
                + " ../utils/container_manager.py -o runnew " \
                + " -c " + container_name + " -i " + image_name + " -a \"" \
                + run_args + "\""
    RUN_LOGGER.debug("Run cmd in the cluster to start container: " + start_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(start_cmd, nnodes, 600)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't start docker container: " +
                         ",".join(bad_hosts.keys()))
        return False
    return True


def stop_container_in_cluster(dp_path, container_name, nnodes):
    '''Call CLUSTER_MGR tool to stop containers.'''
    stop_cont_cmd = "cd " + dp_path + " && " + sys.executable \
                    + " ../utils/container_manager.py -o stop" \
                    + " -c " + container_name
    RUN_LOGGER.debug("Run cmd to stop container(s) in the cluster:" +
                     stop_cont_cmd)
    failed_hosts = CLUSTER_MGR.run_command_some_hosts(stop_cont_cmd, nnodes,
                                                      60)
    if len(failed_hosts) != 0:
        RUN_LOGGER.warning("Hosts that stop container " + container_name +
                           " failed:" + ",".join(failed_hosts.keys()) +
                           " Continue.")
        return False
    RUN_LOGGER.info("All containers stoped in the cluster")
    return True


def clear_caches_cluster(clear, nnodes):
    '''Set vm.drop to clean the system caches.'''
    if not clear:
        RUN_LOGGER.info("Caches clear config is NOT set.")
        return

    clear_cmd = "sync && sudo /sbin/sysctl vm.drop_caches=3"
    timeout = 30
    RUN_LOGGER.debug("Run cmd in the cluster to clear the system cache: " +
                     clear_cmd + " timeout=" + str(timeout))
    failed_hosts = CLUSTER_MGR.run_command_some_hosts(clear_cmd, nnodes,
                                                      timeout)
    if len(failed_hosts) != 0:
        RUN_LOGGER.warning("Hosts that clear cache failed: " +
                           ",".join(failed_hosts.keys()) + ". Continue.")
    RUN_LOGGER.info("Clear system caches if it set......[SUCCESS]")


def start_monitors_in_cluster(dp_path, case_log_dir, nnodes, config):
    '''Start sytem and vendor's monitors.'''
    start_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                    + " ../utils/sys_monitor.py -o restart -l "
    timeout = 60
    RUN_LOGGER.debug("Run cmd in the cluster to start system monitors: " +
                     start_mon_cmd)
    bad_hosts = CLUSTER_MGR.start_monitors_some_hosts(start_mon_cmd,
                                                      case_log_dir, nnodes,
                                                      timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't start system monitors: " +
                         ",".join(bad_hosts.keys()))

    ven_mon_path = os.path.join(dp_path, "vendors", config.VENDOR,
                                config.VENDOR + "_monitor.py")
    start_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                    + " " + ven_mon_path + " -o restart -l "
    RUN_LOGGER.debug("Run cmd in the cluster to start vendor's monitors: " +
                     start_mon_cmd)
    bad_hosts = CLUSTER_MGR.start_monitors_some_hosts(start_mon_cmd,
                                                      case_log_dir, nnodes,
                                                      timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't start vendor's monitors: " +
                         ",".join(bad_hosts.keys()))


def stop_monitors_in_cluster(dp_path, nnodes, config):
    '''Stop sytem and vendor's monitors.'''
    stop_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                   + " ../utils/sys_monitor.py -o stop"
    timeout = 60
    RUN_LOGGER.debug("Run cmd in the cluster to stop system monitors: " +
                     stop_mon_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(stop_mon_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't stop system monitors: " +
                         ",".join(bad_hosts.keys()))

    ven_mon_path = os.path.join(dp_path, "vendors", config.VENDOR,
                                config.VENDOR + "_monitor.py")
    stop_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                   + " " + ven_mon_path + " -o stop"
    RUN_LOGGER.debug("Run cmd in the cluster to stop vendor's monitors: " +
                     stop_mon_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(stop_mon_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't stop vendor's monitors: " +
                         ",".join(bad_hosts.keys()))


def start_tasks_in_cluster(dp_path, container_name, config, base_args,
                           curr_log_path, case):
    '''Start tasks in cluster, and NOT wait.'''
    nnodes = len(config.HOSTS)
    framework = config.CASES[case]

    op, df, spectflops, oplib, chip = case.split(":")
    env_dir = os.path.join(config.FLAGPERF_PATH, "benchmarks", op,
                           config.VENDOR, chip)

    env_shell = os.path.join(env_dir, "env.sh")
    req_file = os.path.join(env_dir, "requirements.txt")

    abs_log_path = os.path.join(dp_path, curr_log_path)

    start_cmd = "cd " + dp_path + " && " + sys.executable \
                + " ../utils/container_manager.py -o runcmdin -c " \
                + container_name + " -d -r \"echo Hello FlagPerf" \
                + " > " + abs_log_path + "/hello.log.txt"

    if os.path.isfile(req_file):
        start_cmd += " && pip install -r " + req_file \
                     + " > " + abs_log_path + "/pip_install.log.txt " \
                     + "2>&1"

    if os.path.isfile(env_shell):
        start_cmd += " && source " + env_shell \
                     + " > " + abs_log_path + "/env.log.txt " \
                     + "2>&1"

    start_cmd += " && python3 " + config.FLAGPERF_PATH + "/container_main.py" + base_args \
                 + " > " + abs_log_path + "/container_main.log.txt " \
                 + "2>&1"

    start_cmd += " \""

    RUN_LOGGER.debug("Run cmd in the cluster to start tasks, cmd=" + start_cmd)
    CLUSTER_MGR.run_command_some_hosts_distribution_info(
        start_cmd, nnodes, 15, "base")
    # Wait a moment for starting tasks.
    time.sleep(60)


def wait_for_finish(dp_path, container_name, pid_file_path, nnodes):
    '''wait all the processes of start_xxx_task.py finished.
    '''
    # Aussme pid of start_xxx_task.py won't loop in a short time.
    check_cmd = "cd " + dp_path + "; " + sys.executable \
                + " ../utils/container_manager.py -o pidrunning -c " \
                + container_name + " -f " + pid_file_path

    RUN_LOGGER.debug(
        "Run cmd to check whether the training tasks is running: " + check_cmd)
    while True:
        bad_hosts = CLUSTER_MGR.run_command_some_hosts(check_cmd,
                                                       nnodes,
                                                       no_log=True)
        if len(bad_hosts) == nnodes:
            break
        time.sleep(10)


def prepare_containers_env_cluster(dp_path, case_log_dir, container_name,
                                   image_name, nnodes, config):
    '''Prepare containers environments in the cluster. It will start
       containers, setup environments, start monitors, and clear caches.'''
    container_start_args = " --rm --init --detach --net=host --uts=host" \
                           + " --ipc=host --security-opt=seccomp=unconfined" \
                           + " --privileged=true --ulimit=stack=67108864" \
                           + " --ulimit=memlock=-1" \
                           + " -w " + config.FLAGPERF_PATH \
                           + " --shm-size=" + config.SHM_SIZE \
                           + " -v " + dp_path + ":" \
                           + config.FLAGPERF_PATH

    if config.ACCE_CONTAINER_OPT is not None:
        container_start_args += " " + config.ACCE_CONTAINER_OPT

    RUN_LOGGER.info("a) Stop old container(s) first.")
    stop_container_in_cluster(dp_path, container_name, nnodes)
    RUN_LOGGER.info("b) Start container(s) in the cluster.")
    if not start_container_in_cluster(dp_path, container_start_args,
                                      container_name, image_name, nnodes):
        RUN_LOGGER.error("b) Start container in the cluster......"
                         "[FAILED]. Ignore this round.")
        return False
    RUN_LOGGER.info("b) Start container(s) in the cluster.......[SUCCESS]")

    RUN_LOGGER.info("c) Start monitors......")
    start_monitors_in_cluster(dp_path, case_log_dir, nnodes, config)
    RUN_LOGGER.info("d) Clear system caches if it set......")
    clear_caches_cluster(config.CLEAR_CACHES, nnodes)
    return True


def clean_containers_env_cluster(dp_path, container_name, nnodes, config):
    '''Clean containers environments in the cluster. It will stop containers,
       and stop monitors.'''
    RUN_LOGGER.info("a) Stop containers......")
    stop_container_in_cluster(dp_path, container_name, nnodes)
    RUN_LOGGER.info("b) Stop monitors......")
    stop_monitors_in_cluster(dp_path, nnodes, config)


def get_valid_cases(config):
    '''Check case config in test_conf, return valid cases list.'''
    if not isinstance(config.CASES, dict):
        RUN_LOGGER.error(
            "No valid cases found in config/host.yaml because config.CASES is not a dict...[EXIT]"
        )
        sys.exit(4)
    RUN_LOGGER.debug("Check configs of all test cases: " +
                     ",".join(config.CASES))
    valid_cases = []
    cases_config_error = []
    for case in config.CASES:
        valid_cases.append(case)
    if len(valid_cases) == 0:
        RUN_LOGGER.error("No valid cases found in config/host.yaml...[EXIT]")
        sys.exit(4)
    RUN_LOGGER.debug("Valid cases: " + ",".join(valid_cases))
    RUN_LOGGER.info("Get valid cases list......[SUCCESS]")
    return valid_cases


def collect_and_merge_logs(curr_log_path, cases, nnodes):
    '''Scp logs from hosts in the cluster to temp dir, and then merge all.
    '''
    get_all = True
    RUN_LOGGER.info("Collect logs in cluster.")
    for case in cases:
        case_log_dir = os.path.join(curr_log_path, case)
        RUN_LOGGER.debug("Case " + case + ", log dir: " + case_log_dir)
        failed_hosts = CLUSTER_MGR.collect_files_some_hosts(curr_log_path,
                                                            curr_log_path,
                                                            nnodes,
                                                            timeout=600)
        if len(failed_hosts) != 0:
            RUN_LOGGER.error("Case " + case + ", log dir: " + case_log_dir +
                             " collect log failed on hosts: " +
                             ",".join(failed_hosts))
            get_all = False
        else:
            RUN_LOGGER.info("Case " + case + ", get all logs in dir: " +
                            case_log_dir)

    if get_all:
        RUN_LOGGER.info("Congrats! See all logs in " + curr_log_path)
    else:
        RUN_LOGGER.warning("Sorry! Not all logs have been collected in " +
                           curr_log_path)


def summary_logs(config, case_log_dir):
    analysis_module_path = os.path.join("vendors", config.VENDOR,
                                        config.VENDOR + "_analysis")
    analysis_module_path = analysis_module_path.replace("/", ".")
    analysis_module = importlib.import_module(analysis_module_path)
    analysis_log = getattr(analysis_module, 'analysis_log', None)

    result = {}
    noderank = 0
    for host in config.HOSTS:
        result[host] = {}
        monitor_log_dir = os.path.join(case_log_dir,
                                       host + "_noderank" + str(noderank))

        # vendor monitor results like temp/power
        vendor_monitor_path = os.path.join(monitor_log_dir,
                                           config.VENDOR + "_monitor.log")
        vendor_log = analysis_log(vendor_monitor_path, config)
        result[host]["vendor"] = vendor_log

        # system monitor results like CPU/MEM/POWER
        for index in ["cpu", "mem", "pwr"]:
            monitor_path = os.path.join(monitor_log_dir,
                                        index + "_monitor.log")
            with open(monitor_path, 'r') as file:
                sys_log = [
                    float(line.split("\t")[1][:-1]) for line in file
                    if "\t" in line
                ]
            result[host][index] = sys_log

        # FlagPerf Result
        flagperf_result_path = os.path.join(monitor_log_dir,
                                            "operation.log.txt")
        with open(flagperf_result_path, 'r') as file:
            key_lines = [
                line.strip() for line in file if 'FlagPerf Result' in line
            ]
        result[host]["flagperf"] = key_lines

        noderank += 1

    return result


def analysis_log(key_logs):
    noderank = 0
    for host in key_logs:
        RUN_LOGGER.info("*" * 50)
        RUN_LOGGER.info("Noderank {} with IP {}".format(noderank, host))

        RUN_LOGGER.info("1) Performance:")
        for line in key_logs[host]["flagperf"]:
            RUN_LOGGER.info("  " + line.split("]")[1])

        RUN_LOGGER.info("2) POWER:")
        RUN_LOGGER.info("  2.1) SYSTEM POWER:")
        pwr_series = key_logs[host]["pwr"]
        RUN_LOGGER.info(
            "    AVERAGE: {} Watts, MAX: {} Watts, STD DEVIATION: {} Watts".
            format(round(np.mean(pwr_series), 2), round(np.max(pwr_series), 2),
                   round(np.std(pwr_series), 2)))

        RUN_LOGGER.info("  2.2) AI-chip POWER:")
        for node in key_logs[host]["vendor"]["power"].keys():
            pwr_series = key_logs[host]["vendor"]["power"][node]
            kmeans_series = []
            for item in pwr_series:
                if (np.max(pwr_series) - item) <= (item - np.min(pwr_series)):
                    kmeans_series.append(item)
            pwr_series = kmeans_series
            RUN_LOGGER.info(
                "    RANK {}'s AVERAGE: {} Watts, MAX: {} Watts, STD DEVIATION: {} Watts"
                .format(node, round(np.mean(pwr_series), 2),
                        round(np.max(pwr_series), 2),
                        round(np.std(pwr_series), 2)))

        RUN_LOGGER.info("  2.3) AI-chip TEMPERATURE:")
        for node in key_logs[host]["vendor"]["temp"].keys():
            temp_series = key_logs[host]["vendor"]["temp"][node]
            kmeans_series = []
            for item in temp_series:
                if (np.max(temp_series) - item) <= (item -
                                                    np.min(temp_series)):
                    kmeans_series.append(item)
            temp_series = kmeans_series
            RUN_LOGGER.info(
                u"    RANK {}'s AVERAGE: {} \u00b0C, MAX: {} \u00b0C, STD DEVIATION: {} \u00b0C"
                .format(node, round(np.mean(temp_series), 2),
                        round(np.max(temp_series), 2),
                        round(np.std(temp_series), 2)))

        RUN_LOGGER.info("3) Utilization:")
        RUN_LOGGER.info("  3.1) SYSTEM CPU:")
        cpu_series = key_logs[host]["cpu"]
        RUN_LOGGER.info(
            "    AVERAGE: {} %, MAX: {} %, STD DEVIATION: {} %".format(
                round(np.mean(cpu_series) * 100, 3),
                round(np.max(cpu_series) * 100, 3),
                round(np.std(cpu_series) * 100, 3)))

        RUN_LOGGER.info("  3.2) SYSTEM MEMORY:")
        mem_series = key_logs[host]["mem"]
        RUN_LOGGER.info(
            "    AVERAGE: {} %, MAX: {} %, STD DEVIATION: {} %".format(
                round(np.mean(mem_series) * 100, 3),
                round(np.max(mem_series) * 100, 3),
                round(np.std(mem_series) * 100, 3)))

        RUN_LOGGER.info("  3.3) AI-chip MEMORY:")
        for node in key_logs[host]["vendor"]["mem"].keys():
            mem_series = key_logs[host]["vendor"]["mem"][node]
            RUN_LOGGER.info(
                "    RANK {}'s AVERAGE: {} %, MAX: {} %, STD DEVIATION: {} %".
                format(
                    node,
                    round(
                        np.mean(mem_series) * 100 /
                        key_logs[host]["vendor"]["max_mem"], 3),
                    round(
                        np.max(mem_series) * 100 /
                        key_logs[host]["vendor"]["max_mem"], 3),
                    round(
                        np.std(mem_series) * 100 /
                        key_logs[host]["vendor"]["max_mem"], 3)))
        noderank += 1


def print_welcome_msg():
    '''Print colorful welcome message to console.'''
    print("\033[1;34;40m==============================================\033[0m")
    print("\033[1;36;40m          Welcome to FlagPerf!\033[0m")
    print(
        "\033[1;36;40m      See more at https://github.com/FlagOpen/FlagPerf \033[0m"
    )
    print("\033[1;34;40m==============================================\033[0m")


def log_test_configs(cases, curr_log_path, dp_path, config):
    '''Put test configs to log '''
    RUN_LOGGER.info("--------------------------------------------------")
    RUN_LOGGER.info("Prepare to run flagperf benchmakrs with configs: ")
    RUN_LOGGER.info("Deploy path on host:\t" + dp_path)
    RUN_LOGGER.info("Vendor:\t\t" + config.VENDOR)
    RUN_LOGGER.info("Testcases:\t\t[" + ','.join(cases) + "]")
    RUN_LOGGER.info("Log path on host:\t" + curr_log_path)
    RUN_LOGGER.info("Cluster:\t\t[" + ",".join(config.HOSTS) + "]")
    RUN_LOGGER.info("--------------------------------------------------")


def main():
    '''Main process to run all the testcases'''

    print_welcome_msg()

    # load yaml
    with open("configs/host.yaml", "r") as file:
        config_dict = yaml.safe_load(file)
        config = Namespace(**config_dict)

    # Set logger first
    timestamp_log_dir = "run" + time.strftime("%Y%m%d%H%M%S", time.localtime())
    curr_log_path = os.path.join(config.FLAGPERF_LOG_PATH, timestamp_log_dir)
    RUN_LOGGER.init(curr_log_path,
                    "flagperf_run.log",
                    config.FLAGPERF_LOG_LEVEL,
                    "both",
                    log_caller=True)

    RUN_LOGGER.info("======== Step 1: Check environment and configs. ========")
    RUN_LOGGER.info("Initialize logger with log path: " + curr_log_path +
                    "......[SUCCESS]")

    # Check test environment and configs of testcases.
    CLUSTER_MGR.init(config.HOSTS,
                     config.SSH_PORT,
                     getpass.getuser(),
                     logger=RUN_LOGGER)
    check_cluster_health()
    dp_path = os.path.abspath(config.FLAGPERF_PATH)
    check_cluster_deploy_path(dp_path)
    cases = get_valid_cases(config)
    log_test_configs(cases, curr_log_path, dp_path, config)

    RUN_LOGGER.info("========= Step 2: Prepare and Run test cases. =========")

    for case in cases:
        RUN_LOGGER.info("======= Testcase: " + case + " =======")

        framework = config.CASES[case]

        # Prepare docker image.
        image_mgr = image_manager.ImageManager(
            "flagperf-operation-" + config.VENDOR + "-" + framework,
            "t_" + VERSION)
        image_name = image_mgr.repository + ":" + image_mgr.tag
        nnodes = len(config.HOSTS)
        RUN_LOGGER.info("=== 2.1 Prepare docker image:" + image_name + " ===")
        if not prepare_docker_image_cluster(dp_path, image_mgr, framework,
                                            nnodes, config):
            RUN_LOGGER.error("=== 2.1 Prepare docker image...[FAILED] " +
                             "Ignore this case " + case + " ===")
            continue

        # Set command to start docker container in the cluster
        container_name = image_mgr.repository + "-" + image_mgr.tag \
                         + "-container"

        # Set command to start train script in container in the cluster
        log_dir_container = os.path.join(config.FLAGPERF_LOG_PATH,
                                         timestamp_log_dir)
        base_args = " --vendor " + config.VENDOR + " --case_name " + case \
                    + " --nnodes " + str(nnodes) \
                    + " --perf_path " + dp_path \
                    + " --nproc_per_node " + str(config.NPROC_PER_NODE) \
                    + " --log_dir " + os.path.join(dp_path, log_dir_container) \
                    + " --log_level " + config.FLAGPERF_LOG_LEVEL.upper() \
                    + " --master_port " + config.MASTER_PORT

        RUN_LOGGER.info("=== 2.2 Setup container and run testcases. ===")

        RUN_LOGGER.info("-== Testcase " + case + " starts ==-")
        RUN_LOGGER.info("1) Prepare container environments in cluster...")
        case_log_dir = os.path.join(curr_log_path, case)
        if not prepare_containers_env_cluster(
                dp_path, case_log_dir, container_name, image_name, nnodes,
                config):
            RUN_LOGGER.error("1) Prepare container environments in cluster"
                             "...[FAILED]. Ignore case " + case)
            continue
        RUN_LOGGER.info("2) Start tasks in the cluster...")

        start_tasks_in_cluster(dp_path, container_name, config, base_args,
                               curr_log_path, case)

        # Wait until start_xxx_task.py finished.
        RUN_LOGGER.info("3) Waiting for tasks end in the cluster...")
        pid_file_path = os.path.join(log_dir_container, "start_base_task.pid")
        wait_for_finish(dp_path, container_name, pid_file_path, nnodes)

        RUN_LOGGER.info("3) Training tasks end in the cluster...")
        RUN_LOGGER.info("4) Clean container environments in cluster...")
        clean_containers_env_cluster(dp_path, container_name, nnodes, config)
        RUN_LOGGER.info("-== Testcase " + case + " finished ==-")
        RUN_LOGGER.info("=== 2.2 Setup container and run testcases finished."
                        " ===")
    RUN_LOGGER.info("========= Step 3: Collect logs in the cluster. =========")
    RUN_LOGGER.info("1) merge logs from all nodes to master")
    collect_and_merge_logs(os.path.join(dp_path, curr_log_path), cases, nnodes)

    RUN_LOGGER.info("2) summary logs")
    key_logs = summary_logs(config, case_log_dir)
    RUN_LOGGER.debug(key_logs)
    jsonfile = os.path.join(dp_path, curr_log_path, "detail_result.json")
    json.dump(key_logs, open(jsonfile, "w"))

    RUN_LOGGER.info("3) analysis logs")
    analysis_log(key_logs)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        usage()
    main()
    RUN_LOGGER.stop()

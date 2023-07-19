# Copyright (c) 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
''' TODO Copyright and Other info '''

import os
import sys
import time
import getpass
from config import cluster_conf as cc
from config import test_conf as tc

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
    print("Edit config file test_conf.py & cluster_conf.py in "
          "training/run_benchmarks/config and run.")
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


def _get_deploy_path():
    '''Return deploy path according to FLAGPERF_LOG_PATH_HOST in test_conf.'''
    if 'FLAGPERF_PATH' not in tc.__dict__.keys() \
       or tc.FLAGPERF_PATH is None:
        dp_path = os.path.abspath(os.path.join(CURR_PATH, "../../training/"))
    else:
        dp_path = os.path.abspath(tc.FLAGPERF_PATH)
    return dp_path


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


def check_testconf():
    ''' Check test config.
        Make sure all CASES are configed.
    '''
    RUN_LOGGER.debug("Check test config: VENDOR")
    if 'VENDOR' not in tc.__dict__.keys():
        RUN_LOGGER.error("VENDOR MUST be set in test_conf...[EXIT]")
        sys.exit(2)
    RUN_LOGGER.info("Check test config: VENDOR......[SUCCESS]")


def check_case_config(case, case_config, vendor):
    '''Check config of the testcase. Make sure its path exists, framework is
       right and config file exists.
    '''
    RUN_LOGGER.debug("Check config of test case: " + case)
    must_configs = [
        "model", "framework", "nnodes", "nproc", "config", "repeat",
        "data_dir_host", "data_dir_container"
    ]
    for config_item in case_config.keys():
        if config_item in must_configs:
            must_configs.remove(config_item)
    if len(must_configs) > 0:
        RUN_LOGGER.warning("Case " + case + " misses some config items: " +
                           ",".join(must_configs))
        return False

    framework = case_config["framework"].split("_")[0]
    model_path = CURR_PATH + "/../benchmarks/" + case_config["model"] \
        + "/" + framework
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        RUN_LOGGER.warning("Case " + case + ": deploy path doesn't exist: " +
                           model_path)
        return False

    config_path = CURR_PATH + "/../" + vendor + "/" + case_config["model"] + \
        "-" + framework + "/config/" + \
        case_config["config"] + ".py"
    if not os.path.isfile(config_path):
        RUN_LOGGER.warning("Case " + case + ": config file doesn't exist: " +
                           config_path)
        return False
    nnodes = case_config["nnodes"]
    cluster_host_counts = CLUSTER_MGR.get_hosts_count()
    # TODO Check nprocs < 8?
    if nnodes > cluster_host_counts:
        RUN_LOGGER.error("This case seems need more hosts than cluster has. " +
                         "The count of need hosts is " + str(nnodes) +
                         ", but cluster has " + str(cluster_host_counts))
        return False

    RUN_LOGGER.debug("Check config of test case: " + case + " ...[SUCCESS]")
    return True


def prepare_docker_image_cluster(dp_path, image_mgr, framework, nnodes):
    '''Prepare docker image in registry and in the cluster.
    '''
    vendor = tc.VENDOR
    image_vendor_dir = os.path.join(
        CURR_PATH, "../" + vendor + "/docker_image/" + framework)
    image_name = image_mgr.repository + ":" + image_mgr.tag
    RUN_LOGGER.debug("Prepare docker image in cluster. image_name=" +
                     image_name + " image_vendor_dir=" + image_vendor_dir)
    prepare_image_cmd = "cd " + dp_path + " && " + sys.executable \
                        + " utils/image_manager.py -o build -i " \
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


def prepare_running_env(dp_path, container_name, case_config, stdout, nullout):
    '''Install extensions and setup env before start task in container.
    '''
    nnodes = case_config["nnodes"]
    model = case_config["model"]
    framework = case_config["framework"]
    prepare_cmd = "cd " + dp_path + " && " + sys.executable \
                  + " utils/container_manager.py -o runcmdin -c " \
                  + container_name + " -t 1800 -r \"python3 " \
                  + tc.FLAGPERF_PATH + "/" \
                  + "/run_benchmarks/prepare_in_container.py --framework " \
                  + framework + " --model " + model + " --vendor " \
                  + tc.VENDOR + " --pipsource " + tc.PIP_SOURCE + "\""
    pre_env_cmd = "sudo docker exec -i " + container_name + " bash -c \"" + "python3 " \
                  + tc.FLAGPERF_PATH + "/" \
                  + "/run_benchmarks/prepare_in_container.py --framework " \
                  + framework + " --model " + model + " --vendor " \
                  + tc.VENDOR + " --pipsource " + tc.PIP_SOURCE + "\""
    sys.stdout = stdout
    print("Command 2, run at host")
    print("    " + pre_env_cmd)
    print("Command 3: run at host")
    print("[INFO] Command 3 let you go into docker(container)")
    print("    sudo docker exec -it " + container_name + " /bin/bash")
    sys.stdout = nullout
    timeout = 1800
    RUN_LOGGER.debug(
        "Run cmd in the cluster to prepare running environment: " +
        prepare_cmd + " timeout=" + str(timeout))
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(prepare_cmd, nnodes,
                                                   timeout)

    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't prepare running environment " +
                         "properly: " + ",".join(bad_hosts.keys()))
        return False
    return True


def start_container_in_cluster(dp_path, run_args, container_name, image_name,
                               nnodes, stdout, nullout):
    '''Call CLUSTER_MGR tool to start containers.'''
    start_cmd = "cd " + dp_path + " && " + sys.executable \
                + " utils/container_manager.py -o runnew " \
                + " -c " + container_name + " -i " + image_name + " -a \"" \
                + run_args + "\""
    start_container_cmd = "sudo docker run " + run_args + \
                      " --name=" + container_name + " \"" + image_name + "\" " + \
                      "sleep infinity"
    sys.stdout = stdout
    print("You should run these 4 commands to start the task manually:")
    print("Command 1: run at host")
    print("    " + start_container_cmd)
    sys.stdout = nullout
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
                    + " utils/container_manager.py -o stop" \
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


def start_monitors_in_cluster(dp_path, case_log_dir, nnodes):
    '''Start sytem and vendor's monitors.'''
    start_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                    + " utils/sys_monitor.py -o restart -l "
    timeout = 60
    RUN_LOGGER.debug("Run cmd in the cluster to start system monitors: " +
                     start_mon_cmd)
    bad_hosts = CLUSTER_MGR.start_monitors_some_hosts(start_mon_cmd,
                                                      case_log_dir, nnodes,
                                                      timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't start system monitors: " +
                         ",".join(bad_hosts.keys()))

    ven_mon_path = os.path.join(dp_path, tc.VENDOR, tc.VENDOR + "_monitor.py")
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


def stop_monitors_in_cluster(dp_path, nnodes):
    '''Stop sytem and vendor's monitors.'''
    stop_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                   + " utils/sys_monitor.py -o stop"
    timeout = 60
    RUN_LOGGER.debug("Run cmd in the cluster to stop system monitors: " +
                     stop_mon_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(stop_mon_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't stop system monitors: " +
                         ",".join(bad_hosts.keys()))

    ven_mon_path = os.path.join(dp_path, tc.VENDOR, tc.VENDOR + "_monitor.py")
    stop_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                   + " " + ven_mon_path + " -o stop"
    RUN_LOGGER.debug("Run cmd in the cluster to start vendor's monitors: " +
                     stop_mon_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(stop_mon_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        RUN_LOGGER.error("Hosts that can't start vendor's monitors: " +
                         ",".join(bad_hosts.keys()))


def start_tasks_in_cluster(dp_path, container_name, case_config, base_args,
                           count, stdout, nullout, curr_log_path):
    '''Start tasks in cluster, and NOT wait.'''
    framework = case_config["framework"].split("_")[0]
    nnodes = case_config["nnodes"]
    env_file = os.path.join(
        tc.FLAGPERF_PATH, tc.VENDOR,
        case_config["model"] + "-" + case_config["framework"],
        "config/environment_variables.sh")
    if (os.path.isfile(env_file)):
        start_cmd = "cd " + dp_path + " && " + sys.executable \
                + " utils/container_manager.py -o runcmdin -c " \
                + container_name + " -d -r \"source " + env_file \
                + " > " + curr_log_path + "/source_env.log.txt " \
                + "2>&1 && " \
                + "python3 " + tc.FLAGPERF_PATH + "/run_benchmarks/" \
                + framework + "/start_" + framework + "_task.py " \
                + base_args + " --round " + str(count)
    else:
        start_cmd = "cd " + dp_path + " && " + sys.executable \
                + " utils/container_manager.py -o runcmdin -c " \
                + container_name + " -d -r \"" \
                + "python3 " + tc.FLAGPERF_PATH + "/run_benchmarks/" \
                + framework + "/start_" + framework + "_task.py " \
                + base_args + " --round " + str(count)
    if tc.ACCE_VISIBLE_DEVICE_ENV_NAME is not None:
        start_cmd += " --visible_dev_env " \
                     + tc.ACCE_VISIBLE_DEVICE_ENV_NAME
    start_cmd += " \""
    if (os.path.isfile(env_file)):
        start_task_cmd = "source " + env_file + " && " \
                + "python3 " + tc.FLAGPERF_PATH + "/run_benchmarks/" \
                + framework + "/start_" + framework + "_task.py " \
                + base_args + " --round " + str(count)
    else:
        start_task_cmd = "python3 " + tc.FLAGPERF_PATH + "/run_benchmarks/" \
                + framework + "/start_" + framework + "_task.py " \
                + base_args + " --round " + str(count)
    if tc.ACCE_VISIBLE_DEVICE_ENV_NAME is not None:
        start_task_cmd += " --visible_dev_env " \
                     + tc.ACCE_VISIBLE_DEVICE_ENV_NAME
    sys.stdout = stdout
    print("Command 4, run at docker(container)")
    print(
        "[INFO] If you set nnodes != 1, you should run command 1-3 on each hosts, then run the corresponding command 4 respectively"
    )
    hosts = cc.HOSTS
    command_master_ip = start_task_cmd + '  --master_addr ' + hosts[0]
    for i in range(0, nnodes):
        host = hosts[i]
        command = command_master_ip + ' --node_rank ' + str(i) \
                                    + ' --host_addr ' + host
        print("    Command 4 at host " + host + ":")
        print("        " + command)
    sys.stdout = nullout
    RUN_LOGGER.debug("Run cmd in the cluster to start tasks, cmd=" + start_cmd)
    CLUSTER_MGR.run_command_some_hosts_distribution_info(start_cmd, nnodes, 15)
    # Wait a moment for starting tasks.
    time.sleep(60)


def wait_for_finish(dp_path, container_name, pid_file_path, nnodes):
    '''wait all the processes of start_xxx_task.py finished.
    '''
    # Aussme pid of start_xxx_task.py won't loop in a short time.
    check_cmd = "cd " + dp_path + "; " + sys.executable \
                + " utils/container_manager.py -o pidrunning -c " \
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
                                   image_name, case_config, stdout, nullout):
    '''Prepare containers environments in the cluster. It will start
       containers, setup environments, start monitors, and clear caches.'''
    nnodes = case_config["nnodes"]
    container_start_args = " --rm --init --detach --net=host --uts=host" \
                           + " --ipc=host --security-opt=seccomp=unconfined" \
                           + " --privileged=true --ulimit=stack=67108864" \
                           + " --ulimit=memlock=-1" \
                           + " -w " + tc.FLAGPERF_PATH \
                           + " --shm-size=" + tc.SHM_SIZE \
                           + " -v " + dp_path + ":" \
                           + tc.FLAGPERF_PATH \
                           + " -v " + case_config["data_dir_host"] + ":" \
                           + case_config["data_dir_container"]
    if tc.ACCE_CONTAINER_OPT is not None:
        container_start_args += " " + tc.ACCE_CONTAINER_OPT

    RUN_LOGGER.info("a) Stop old container(s) first.")
    stop_container_in_cluster(dp_path, container_name, nnodes)
    RUN_LOGGER.info("b) Start container(s) in the cluster.")
    if not start_container_in_cluster(dp_path, container_start_args,
                                      container_name, image_name, nnodes,
                                      stdout, nullout):
        RUN_LOGGER.error("b) Start container in the cluster......"
                         "[FAILED]. Ignore this round.")
        return False
    RUN_LOGGER.info("b) Start container(s) in the cluster.......[SUCCESS]")

    RUN_LOGGER.info("c) Prepare running environment.")
    if not prepare_running_env(dp_path, container_name, case_config, stdout,
                               nullout):
        RUN_LOGGER.error("c) Prepare running environment......"
                         "[FAILED]. Ignore this round.")
        RUN_LOGGER.info("Stop containers in cluster.")
        stop_container_in_cluster(dp_path, container_name, nnodes)
        return False
    RUN_LOGGER.info("c) Prepare running environment......[SUCCESS]")
    RUN_LOGGER.info("d) Start monitors......")
    start_monitors_in_cluster(dp_path, case_log_dir, nnodes)
    RUN_LOGGER.info("e) Clear system caches if it set......")
    clear_caches_cluster(tc.CLEAR_CACHES, nnodes)
    return True


def clean_containers_env_cluster(dp_path, container_name, nnodes):
    '''Clean containers environments in the cluster. It will stop containers,
       and stop monitors.'''
    RUN_LOGGER.info("a) Stop containers......")
    stop_container_in_cluster(dp_path, container_name, nnodes)
    RUN_LOGGER.info("b) Stop monitors......")
    stop_monitors_in_cluster(dp_path, nnodes)


def collect_and_merge_logs(curr_log_path, cases):
    '''Scp logs from hosts in the cluster to temp dir, and then merge all.
    '''
    get_all = True
    RUN_LOGGER.info("Collect logs in cluster.")
    for case in cases:
        rets, case_config = get_config_from_case(case)
        repeat = case_config["repeat"]
        for i in range(1, repeat + 1):
            case_log_dir = os.path.join(curr_log_path, case, "round" + str(i))
            RUN_LOGGER.debug("Case " + case + ", round " + str(i) +
                             ", log dir: " + case_log_dir)
            nnodes = case_config["nnodes"]
            failed_hosts = CLUSTER_MGR.collect_files_some_hosts(curr_log_path,
                                                                curr_log_path,
                                                                nnodes,
                                                                timeout=600)
            if len(failed_hosts) != 0:
                RUN_LOGGER.error("Case " + case + ", round " + str(i) +
                                 ", log dir: " + case_log_dir +
                                 " collect log failed on hosts: " +
                                 ",".join(failed_hosts))
                get_all = False
            else:
                RUN_LOGGER.info("Case " + case + ", round " + str(i) +
                                ", get all logs in dir: " + case_log_dir)

    if get_all:
        RUN_LOGGER.info("Congrats! See all logs in " + curr_log_path)
    else:
        RUN_LOGGER.warning("Sorry! Not all logs have been collected in " +
                           curr_log_path)


def get_config_from_case(case):
    '''check case is string'''
    if not isinstance(case, str):
        RUN_LOGGER.error("Key in test_config.CASES must be str")
        return False, None

    case_info = case.split(":")
    '''check if 4+ : in case, we don't care what to put in'''
    if len(case_info) < 6:
        RUN_LOGGER.error(
            "At least 6 terms split by \":\" should in test_config.CASES")
        RUN_LOGGER.error("model:framework:hardware_model:nnodes:nproc:repeat")
        return False, None

    case_model = case_info[0]
    case_framework = case_info[1]
    case_hardware = case_info[2]
    case_nnodes = case_info[3]
    case_nproc = case_info[4]
    case_repeat = case_info[5]

    case_config = {"model": case_model}
    case_config["framework"] = case_framework
    case_config[
        "config"] = "config_" + case_hardware + "x" + case_nnodes + "x" + case_nproc
    case_config["repeat"] = int(case_repeat)
    case_config["nnodes"] = int(case_nnodes)
    case_config["nproc"] = int(case_nproc)
    case_config["data_dir_host"] = tc.CASES[case]
    case_config["data_dir_container"] = tc.CASES[case]
    return True, case_config


def get_valid_cases():
    '''Cehck case config in test_conf, return valid cases list.'''
    if not isinstance(tc.CASES, dict):
        RUN_LOGGER.error(
            "No valid cases found in test_conf because test_config.CASES is not a dict...[EXIT]"
        )
        sys.exit(4)
    RUN_LOGGER.debug("Check configs of all test cases: " + ",".join(tc.CASES))
    valid_cases = []
    cases_config_error = []
    for case in tc.CASES:
        rets, case_config = get_config_from_case(case)
        if (not rets) or (not check_case_config(case, case_config, tc.VENDOR)):
            cases_config_error.append(case)
            continue
        valid_cases.append(case)
    if len(valid_cases) == 0:
        RUN_LOGGER.error("No valid cases found in test_conf...[EXIT]")
        sys.exit(4)
    RUN_LOGGER.debug("Valid cases: " + ",".join(valid_cases))
    RUN_LOGGER.debug("Invalid cases that config is error: " +
                     ",".join(cases_config_error))
    RUN_LOGGER.info("Get valid cases list......[SUCCESS]")
    return valid_cases


def print_welcome_msg():
    '''Print colorful welcome message to console.'''
    print("\033[1;34;40m==============================================\033[0m")
    print("\033[1;36;40m          Welcome to FlagPerf!\033[0m")
    print(
        "\033[1;36;40m      See more at https://github.com/FlagOpen/FlagPerf \033[0m"
    )
    print("\033[1;34;40m==============================================\033[0m")


def prepare_case_config_cluster(dp_path, case_config, case):
    '''Sync case config files in cluster.'''
    RUN_LOGGER.info("--------------------------------------------------")
    RUN_LOGGER.info("Testcase " + case + " config:")
    for config_item in case_config.keys():
        RUN_LOGGER.info(config_item + ":\t" + str(case_config[config_item]))
    RUN_LOGGER.info("--------------------------------------------------")
    model = case_config["model"]
    framework = case_config["framework"].split("_")[0]
    config_file = case_config["config"] + ".py"
    nnodes = case_config["nnodes"]
    case_config_dir = os.path.join(dp_path, tc.VENDOR, model + "-" + framework,
                                   "config")
    case_config_file = os.path.join(case_config_dir, config_file)
    failed_hosts = CLUSTER_MGR.sync_file_to_some_hosts(case_config_file,
                                                       case_config_dir, nnodes)
    if len(failed_hosts) != 0:
        RUN_LOGGER.error("Hosts that sync vendor case config file failed: " +
                         ",".join(failed_hosts.keys()))
        return False
    return True


def log_test_configs(cases, curr_log_path, dp_path):
    '''Put test configs to log '''
    RUN_LOGGER.info("--------------------------------------------------")
    RUN_LOGGER.info("Prepare to run flagperf benchmarks with configs: ")
    RUN_LOGGER.info("Deploy path on host:\t" + dp_path)
    RUN_LOGGER.info("Vendor:\t\t" + tc.VENDOR)
    RUN_LOGGER.info("Testcases:\t\t[" + ','.join(cases) + "]")
    RUN_LOGGER.info("Log path on host:\t" + curr_log_path)
    RUN_LOGGER.info("Cluster:\t\t[" + ",".join(cc.HOSTS) + "]")
    RUN_LOGGER.info("--------------------------------------------------")


def main(stdout, nullout):
    '''Main process to run all the testcases'''

    print_welcome_msg()

    # Set logger first
    timestamp_log_dir = "run" + time.strftime("%Y%m%d%H%M%S", time.localtime())
    curr_log_path = os.path.join(tc.FLAGPERF_LOG_PATH, timestamp_log_dir)
    RUN_LOGGER.init(curr_log_path,
                    "flagperf_run.log",
                    tc.FLAGPERF_LOG_LEVEL,
                    "file",
                    log_caller=True)
    RUN_LOGGER.info("======== Step 1: Check environment and configs. ========")
    RUN_LOGGER.info("Initialize logger with log path: " + curr_log_path +
                    "......[SUCCESS]")

    # Check test environment and configs of testcases.
    CLUSTER_MGR.init(cc.HOSTS,
                     cc.SSH_PORT,
                     getpass.getuser(),
                     logger=RUN_LOGGER)
    check_cluster_health()
    dp_path = _get_deploy_path()
    check_cluster_deploy_path(dp_path)
    check_testconf()
    cases = get_valid_cases()
    log_test_configs(cases, curr_log_path, dp_path)

    RUN_LOGGER.info("========= Step 2: Prepare and Run test cases. =========")

    for case in cases:
        RUN_LOGGER.info("======= Testcase: " + case + " =======")
        rets, case_config = get_config_from_case(case)

        # Prepare docker image.
        image_mgr = image_manager.ImageManager(
            "flagperf-" + tc.VENDOR + "-" + case_config["framework"],
            "t_" + VERSION)
        image_name = image_mgr.repository + ":" + image_mgr.tag
        nnodes = case_config["nnodes"]
        RUN_LOGGER.info("=== 2.1 Prepare docker image:" + image_name + " ===")
        if not prepare_docker_image_cluster(dp_path, image_mgr,
                                            case_config["framework"], nnodes):
            RUN_LOGGER.error("=== 2.1 Prepare docker image...[FAILED] " +
                             "Ignore this case " + case + " ===")
            continue

        # Set command to start docker container in the cluster
        container_name = image_mgr.repository + "-" + image_mgr.tag \
                                              + "-container"

        # Set command to start train script in container in the cluster
        log_dir_container = os.path.join(tc.FLAGPERF_LOG_PATH,
                                         timestamp_log_dir)
        base_args = " --vendor " + tc.VENDOR + " --case_name " + case \
                    + " --model_name " + case_config["model"] \
                    + " --train_script " + "run_pretraining.py" \
                    + " --nnodes " + str(nnodes) \
                    + " --nproc " + str(case_config["nproc"]) \
                    + " --hosts " + ",".join(cc.HOSTS) \
                    + " --hosts_ports " + ",".join(cc.HOSTS_PORTS) \
                    + " --data_dir " + case_config["data_dir_container"] \
                    + " --log_dir " + log_dir_container \
                    + " --log_level " + tc.FLAGPERF_LOG_LEVEL \
                    + " --extern_config_file " + case_config["config"] \
                    + ".py" + " --enable_extern_config " \
                    + " --master_port " + cc.MASTER_PORT
        RUN_LOGGER.info("=== 2.2 Prepare case config in cluster. ===")
        if not prepare_case_config_cluster(dp_path, case_config, case):
            RUN_LOGGER.warning("Prepare case config in cluster...[FAILED]. " +
                               "Ignore case " + case)
            continue
        RUN_LOGGER.info("=== 2.3 Setup container and run testcases. ===")
        for count in range(1, case_config["repeat"] + 1):
            RUN_LOGGER.info("-== Testcase " + case + " Round " + str(count) +
                            " starts ==-")
            RUN_LOGGER.info("1) Prepare container environments in cluster...")
            case_log_dir = os.path.join(curr_log_path, case,
                                        "round" + str(count))
            if not prepare_containers_env_cluster(
                    dp_path, case_log_dir, container_name, image_name,
                    case_config, stdout, nullout):
                RUN_LOGGER.error("1) Prepare container environments in cluster"
                                 "...[FAILED]. Ignore case " + case +
                                 " round " + str(count))
                continue
            RUN_LOGGER.info("2) Start tasks in the cluster...")
            start_tasks_in_cluster(dp_path, container_name, case_config,
                                   base_args, count, stdout, nullout,
                                   curr_log_path)

            # Wait until start_xxx_task.py finished.
            RUN_LOGGER.info("3) Waiting for tasks end in the cluster...")
            pid_file_path = os.path.join(
                log_dir_container,
                "start_" + case_config["framework"] + "_task.pid")
            wait_for_finish(dp_path, container_name, pid_file_path, nnodes)
            RUN_LOGGER.info("3) Training tasks end in the cluster...")
            RUN_LOGGER.info("4) Clean container environments in cluster...")
            clean_containers_env_cluster(dp_path, container_name, nnodes)
            RUN_LOGGER.info("-== Testcase " + case + " Round " + str(count) +
                            " finished ==-")
        RUN_LOGGER.info("=== 2.3 Setup container and run testcases finished."
                        " ===")
    RUN_LOGGER.info("========= Step 3: Collect logs in the cluster. =========")
    collect_and_merge_logs(curr_log_path, cases)


if __name__ == '__main__':
    stdout = sys.stdout
    nullout = open("/dev/null", "w")
    sys.stdout = nullout
    if len(sys.argv) > 1:
        usage()
    main(stdout, nullout)
    sys.stdout = stdout
    RUN_LOGGER.stop()

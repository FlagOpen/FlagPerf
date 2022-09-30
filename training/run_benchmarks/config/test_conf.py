'''Test Configs, including'''
# -*-coding:utf-8 -*-

# Set accelerator's vendor name, e.g. iluvatar, cambricon and kunlun.
# We will run benchmarks in training/<vendor>
VENDOR = "nvidia"
# Accelerator options for docker. TODO FIXME support more accelerators.
ACCE_CONTAINER_OPT = " --gpus all"
# XXX_VISIBLE_DEVICE item name in env
# nvidia use CUDA_VISIBLE_DEVICE and cambricon MLU_VISIBLE_DEVICES
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"

# Set type of benchmarks, default or customized.
# default: run benchmarks in training/benchmarks/
# [NOT SUPPORTED] customized: run benchmarks in training/<vendor>/benchmarks/
TEST_TYPE = "default"

# Set pip source, which will be used in preparing envs in container
PIP_SOURCE = "https://mirrors.aliyun.com/pypi/simple"

# The path that flagperf deploy in the cluster.
# If not set, it will be os.path.dirname(run.py)/../../training/
FLAGPERF_PATH_HOST = "/home/flagperf/training"

# Set the mapping directory of flagperf in container.
FLAGPERF_PATH_CONTAINER = "/workspace/flagperf/training"

# Set log path on the host here.
FLAGPERF_LOG_PATH_HOST = FLAGPERF_PATH_HOST + "/result/"
# Set log path in container here.
FLAGPERF_LOG_PATH_CONTAINER = FLAGPERF_PATH_CONTAINER + "/result/"
# Set log level. It should be 'debug', 'info', 'warning' or 'error'.
FLAGPERF_LOG_LEVEL = 'debug'

# System config
# Share memory size
SHM_SIZE = "32G"
# Clear cache config. Clean system cache before running testcase.
CLEAR_CACHES = True

# Set cases you want to run here.
# cases is a list of case name.
CASES = ['BERT_PADDLE_DEMO_A100_1X8',
         'GLM_TORCH_DEMO_A100_1X8',
         'CPM_TORCH_DEMO_A100_1X8']

# Config each case in a dictionary like these.
# <case name> = {
#     # "Set model name"
#     "model": <model name>
#     # If test_type is default, framework should be pytorch.
#     "framework": "<ai framework>",
#     # Set config module in <vendor>/<model>-<framework>/<config>
#     "config": "<testcase config module>",
#     # Set how many times to run this case in container(s).
#     "repeat": 1,
#     # Set how many hosts to run this case
#     "nnodes": 1,
#     # Set how many processes will run on each host
#     "nproc": 2,
#     # Set data path on host: "/home/data_ckpt/bert/train"
#     "data_dir_host": "<data direcotory on host>",
#     # Set data path in container: /mnt/data/bert/train"
#     "data_dir_container": "<data direcotory in container>",
# }

BERT_PADDLE_DEMO_A100_1X8 = {
    "model": "bert",
    "framework": "paddle",
    "config": "config_A100x1x8",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/bert/train/",
    "data_dir_container": "/mnt/data/bert/train/",
}

GLM_TORCH_DEMO_A100_1X8 = {
    "model": "glm",
    "framework": "pytorch",
    "config": "config_A100x1x8",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/glm/train/",
    "data_dir_container": "/mnt/data/glm/train/",
}

GLM_TORCH_DEMO_A100_2X8 = {
    "model": "glm",
    "framework": "pytorch",
    "config": "config_A100x2x8",
    "repeat": 1,
    "nnodes": 2,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/glm/train/",
    "data_dir_container": "/mnt/data/glm/train/",
}

CPM_TORCH_DEMO_A100_1X8 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x8",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}

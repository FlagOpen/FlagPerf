'''Test Configs, including'''
# -*-coding:utf-8 -*-

# Set accelerator's vendor name, e.g. iluvatar, cambricon and kunlunxin.
# We will run benchmarks in training/<vendor>
VENDOR = "iluvatar"
# Accelerator options for docker. TODO FIXME support more accelerators.
# possible value of ACCE_CONTAINER_OPT are:
#   iluvatar:
#       " -v /lib/modules:/lib/modules "
#   kunlunxin:
#       " --device=/dev/xpu0 --device=/dev/xpu1 --device=/dev/xpu2" + \
#       " --device=/dev/xpu3 --device=/dev/xpu4 --device=/dev/xpu5" + \
#       " --device=/dev/xpu6 --device=/dev/xpu7 --device=/dev/xpuctrl"
#   nvidia:
#       " --gpus all"


ACCE_CONTAINER_OPT = " -v /lib/modules:/lib/modules "

# XXX_VISIBLE_DEVICE item name in env
# possible value of ACCE_VISIBLE_DEVICE_ENV_NAME are:
#   CUDA_VISIBLE_DEVICES for nvidia, iluvatar
#   MLU_VISIBLE_DEVICES for cambricon
#   XPU_VISIBLE_DEVICES for kunlunxin
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"

# Set pip source, which will be used in preparing envs in container
PIP_SOURCE = "https://pypi.tuna.tsinghua.edu.cn/simple"

# The path that flagperf deploy in the cluster.
# If not set, it will be os.path.dirname(run.py)/../../training/
FLAGPERF_PATH_HOST = "/data/yanrui/flagperf/yanrui/FlagPerf/training"

# Set the mapping directory of flagperf in container.
FLAGPERF_PATH_CONTAINER = "/workspace/flagperf/training"

# Set log path on the host here.
FLAGPERF_LOG_PATH_HOST = FLAGPERF_PATH_HOST + "/result/"
# Set log path in container here.
FLAGPERF_LOG_PATH_CONTAINER = FLAGPERF_PATH_CONTAINER + "/result/"
# Set log level. It should be 'debug', 'info', 'warning', or 'error'.
FLAGPERF_LOG_LEVEL = 'debug'

# System config
# Share memory size
SHM_SIZE = "64G"
# Clear cache config. Clean system cache before running testcase.
CLEAR_CACHES = True

# Set the case list you want to run here.
# CASES is a list of case names.
CASES = [
    # 'BERT_PADDLE_DEMO_A100_1X8', 
    # 'GLM_TORCH_DEMO_A100_1X8',
    # 'CPM_TORCH_DEMO_A100_1X8',
    "CPM_TORCH_DEMO_BI100_1X8"
]

# Config each case in a dictionary like this.
BERT_PADDLE_DEMO_A100_1X8 = {  # benchmark case name, one in CASES
    "model": "bert",  # model name
    "framework": "paddle",  # AI framework
    "config":
    "config_A100x1x8",  # config module in <vendor>/<model>-<framework>/<config>
    "repeat": 1,  # How many times to run this case
    "nnodes": 1,  # How many hosts to run this case
    "nproc": 8,  # How many processes will run on each host
    "data_dir_host": "/home/datasets_ckpt/bert/train/",  # Data path on host
    "data_dir_container": "/mnt/data/bert/train/",  # Data path in container
}

GLM_TORCH_DEMO_A100_1X1 = {
    "model": "glm",
    "framework": "pytorch",
    "config": "config_A100x1x1",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 1,
    "data_dir_host": "/home/datasets_ckpt/glm/train/",
    "data_dir_container": "/mnt/data/glm/train/",
}

GLM_TORCH_DEMO_A100_1X2 = {
    "model": "glm",
    "framework": "pytorch",
    "config": "config_A100x1x2",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 2,
    "data_dir_host": "/home/datasets_ckpt/glm/train/",
    "data_dir_container": "/mnt/data/glm/train/",
}

GLM_TORCH_DEMO_A100_1X4 = {
    "model": "glm",
    "framework": "pytorch",
    "config": "config_A100x1x4",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 4,
    "data_dir_host": "/home/datasets_ckpt/glm/train/",
    "data_dir_container": "/mnt/data/glm/train/",
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

CPM_TORCH_DEMO_A100_1X1 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x1",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 1,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}

CPM_TORCH_DEMO_A100_1X2 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x2",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 2,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}

CPM_TORCH_DEMO_A100_1X4 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x4",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 4,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
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

GLM_TORCH_DEMO_R300_1X8 = {
    "model": "glm",
    "framework": "pytorch",
    "config": "config_R300x1x8",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/glm/train/",
    "data_dir_container": "/mnt/data/glm/train/",
}

CPM_TORCH_DEMO_BI100_1X8 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_BI-V100x1x8",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 8,
    "data_dir_host": "/data/yanrui/data/cpm/train",
    "data_dir_container": "/mnt/data/cpm/train/",
}

'''Test Configs, including'''
# -*-coding:utf-8 -*-

# Set accelerator's vendor name, e.g. iluvatar, cambricon, kunlunxin and ascend.
# We will run benchmarks in training/<vendor>
VENDOR = "nvidia"

# Accelerator options for docker. TODO FIXME support more accelerators.
# possible value of ACCE_CONTAINER_OPT are:
#   iluvatar:
#       ' -v /lib/modules:/lib/modules '
#   kunlunxin:
#       " --device=/dev/xpu0 --device=/dev/xpu1 --device=/dev/xpu2" + \
#       " --device=/dev/xpu3 --device=/dev/xpu4 --device=/dev/xpu5" + \
#       " --device=/dev/xpu6 --device=/dev/xpu7 --device=/dev/xpuctrl"
#   nvidia:
#       " --gpus all"
#   ascend:
#       "--device=/dev/davinciX --device=/dev/davinci_manager + \
#        --device=/dev/devmm_svm --device=/dev/hisi_hdc + \
#        -v /usr/local/Ascend/driver -v /usr/local/dcmi -v /usr/local/bin/npu-smi"
ACCE_CONTAINER_OPT = " --gpus all"
# XXX_VISIBLE_DEVICE item name in env
# possible value of ACCE_VISIBLE_DEVICE_ENV_NAME are:
#   CUDA_VISIBLE_DEVICES for nvidia, iluvatar
#   MLU_VISIBLE_DEVICES for cambricon
#   XPU_VISIBLE_DEVICES for kunlunxin
#   ASCEND_VISIBLE_DEVICES for ascend
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"

# Set pip source, which will be used in preparing envs in container
PIP_SOURCE = "https://mirror.baidu.com/pypi/simple"

# The path that flagperf deploy in the cluster.
# Users must set FLAGPERF_PATH to where flagperf deploy
# You can assume the preset "/home/FlagPerf/training" points to Null
FLAGPERF_PATH = "/ssd2/laixinyi/projects/FlagPerf/training"
# Set log path on the host here.
FLAGPERF_LOG_PATH = FLAGPERF_PATH + "/result/"

# Set log level. It should be 'debug', 'info', 'warning', or 'error'.
FLAGPERF_LOG_LEVEL = 'debug'

# System config
# Share memory size
SHM_SIZE = "32G"
# Clear cache config. Clean system cache before running testcase.
CLEAR_CACHES = True

# Set the case dict you want to run here.
'''
# Users must use {
    "model:framework:hardwareID:nnodes:nproc:repeat": "dataset path"}
'''
CASES = {
    # "bert:pytorch_1.8:A100:1:8:1": "/home/datasets_ckpt/bert/train/",
    # "glm:pytorch_1.8:A100:1:8:1": "/home/datasets_ckpt/glm/train/",
    # "cpm:pytorch_1.8:A100:1:8:1": "/home/datasets_ckpt/cpm/train/",

    # "mobilenetv2:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "vit:pytorch_1.13:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "efficientnet:pytorch_1.13:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",

    # "faster_rcnn:pytorch_1.8:A100:1:8:1": "/raid/dataset/fasterrcnn/coco2017/",
    # "bigtransfer:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",

    # "tacotron2:pytorch_1.8:A100:1:8:1": "/raid/dataset/tacotron2/LJSpeech/",
    # "resnet50:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "mask_rcnn:pytorch_1.8:A100:1:8:1": "/raid/dataset/maskrcnn/coco2017",
    
    # "wav2vec2:pytorch_1.13:A100:1:8:1": "/raid/dataset/wav2vec2_data/LibriSpeech",
    # "WaveGlow:pytorch_1.13:A100:1:8:1": "/raid/dataset/LJSpeech/",

    
    # "transformer:pytorch_1.13:A100:1:8:1": "/home/datasets_ckpt/transformer/train/",
    # "swin_transformer:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",

    "llama:paddle_2.5.0:V100:1:8:1": "/ssd2/laixinyi/projects/FlagPerf/training/benchmarks/llama/paddle/data",
}

'''Test Configs, including'''
# -*-coding:utf-8 -*-

# Set accelerator's vendor name, e.g. iluvatar, cambricon, kunlunxin, ascend, mthreads, metax and dcu.
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
#   mthreads:
#       " --env MTHREADS_VISIBLE_DEVICES=all"
#   metax:
#       " --device=/dev/dri --device=/dev/mxcd --group-add video"
#   dcu:
#       "-v /opt/hyhal/:/opt/hyhal/ --device=/dev/kfd --device=/dev/dri/ --group-add video"
ACCE_CONTAINER_OPT = " --gpus all"
# XXX_VISIBLE_DEVICE item name in env
# possible value of ACCE_VISIBLE_DEVICE_ENV_NAME are:
#   CUDA_VISIBLE_DEVICES for nvidia, iluvatar
#   MLU_VISIBLE_DEVICES for cambricon
#   XPU_VISIBLE_DEVICES for kunlunxin
#   ASCEND_VISIBLE_DEVICES for ascend
#   MUSA_VISIBLE_DEVICES for mthreads
#   HIP_VISIBLE_DEVICES for dcu
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"

# Set pip source, which will be used in preparing envs in container
PIP_SOURCE = "https://mirror.baidu.com/pypi/simple"

# The path that flagperf deploy in the cluster.
# Users must set FLAGPERF_PATH to where flagperf deploy
# You can assume the preset "/home/FlagPerf/training" points to Null
FLAGPERF_PATH = "/home/FlagPerf/training"
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
    # nvidia cases
    #"llama3_8B:megatron_core060:A100:1:8:1": "/data/llama3_8b_pretrain"
    # "llama3_70B:megatron_core060:H100:8:8:1": "/data/llama3_70b_pretrain"
    # "bert:pytorch_1.8:A100:1:8:1": "/raid/home_datasets_ckpt/bert/train/",
    # "glm:pytorch_1.8:A100:1:8:1": "/raid/home_datasets_ckpt/glm/train/",
    # "cpm:pytorch_1.8:A100:1:8:1": "/raid/home_datasets_ckpt/cpm/train/",


    # "llava1.5_7b:deepspeed-torch:A800:1:8:1": "/raid/dataset/LLAVA/",
    #"llama2_7b_finetune:pytorch_2.0.1:A100:1:1:1": "/raid/dataset/llama2_finetune/",
    #"aquila2_7b_finetune:flagscale:A800:1:8:1": "/raid/dataset/aquila2_7b_finetune",
    # "llama2_7b_finetune:pytorch_2.0.1:A100:1:1:1": "/raid/dataset/llama2_finetune/",
    # "aquila2_7b_finetune:flagscale:A800:1:8:1": "/raid/dataset/aquila2_7b_finetune",
    # "mobilenetv2:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "vit:pytorch_1.13:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "efficientnet:pytorch_1.13:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",

    # "faster_rcnn:pytorch_1.8:A100:1:8:1": "/raid/dataset/fasterrcnn/coco2017/",
    # "bigtransfer:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",

    # "tacotron2:pytorch_1.13:A100:1:8:1": "/raid/dataset/tacotron2/LJSpeech/",
    # "resnet50:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "mask_rcnn:pytorch_1.8:A100:1:8:1": "/raid/dataset/maskrcnn/coco2017",
    # "dlrm:pytorch_1.10:A100:1:8:1": "/raid/dataset/criteo_1TB_click_logs/binary_dataset/",
    
    # "wav2vec2:pytorch_1.13:A100:1:8:1": "/raid/dataset/wav2vec2_data/LibriSpeech",
    # "WaveGlow:pytorch_1.13:A100:1:8:1": "/raid/dataset/LJSpeech/",
    # "resnet50:tensorflow2:A100:1:8:1": "/raid/dataset/ImageNet2012/tf_records/",
    # "moflow:pytorch_1.13:A100:1:8:1": "/raid/dataset/MoFlow/data/",

    # "distilbert:pytorch_1.12:A100:1:8:1": "/raid/dataset/distilbert/",
    
    # "transformer:pytorch_1.13:A100:1:8:1": "/raid/dataset/transformer/wmt14_en_de_joined_dict",
    # "swin_transformer:pytorch_1.8:A100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "transformer_xl:pytorch_1.8:A100:1:8:1": "/raid/dataset/transformer_xl/",
    # "t5_small:pytorch_1.12:A100:1:8:1": "/raid/dataset/t5_small_train",
    # "gpt2:pytorch_1.12:A100:1:8:1": "/raid/dataset/gpt2",

    # "bert_hf:pytorch_1.13:A100:1:8:1": "/raid/dataset/bert_hf_train",
    # "longformer:pytorch_1.12:A100:1:8:1": "/raid/dataset/longformer_train/",
    # "detr:pytorch_1.13:A100:1:8:1": "/raid/dataset/detr/coco2017/",
    
    # "llama2_7b:deepspeed:A100:1:8:1": "/raid/dataset/llama2_7b_pretrain",
    # "aquila2_7b:flagscale:A100:1:8:1": "/raid/dataset/aquila2_7b_pretrain",
    # "llama2_70B:megatron:H800:4:8:1": "/raid/dataset/llama2_70B_pretrain",
    # "llama1_7B:paddle_2.5.1:TP1PP1SH2SP8A10040G:1:8:1":"/raid/dataset/llama/"
    # "llama1_7B:paddle_2.5.1:TP2PP1SH1SP4A10040G:1:8:1":"/raid/dataset/llama/"
    # "llama1_7B:paddle_2.5.1:TP2PP1SH2SP4A10040G:1:8:1":"/raid/dataset/llama/"
    # "llama1_7B:paddle_2.5.1:TP2PP4SH1SP1A10040G:1:8:1":"/raid/dataset/llama/"
    # "llama1_13B:paddle_2.5.1:TP1PP1SH2SP8A10040G:1:8:1":"/raid/dataset/llama/"
    # "llama1_13B:paddle_2.5.1:TP2PP1SH1SP4A10040G:1:8:1":"/raid/dataset/llama/"
    # "llama1_13B:paddle_2.5.1:TP2PP1SH2SP4A10040G:1:8:1":"/raid/dataset/llama/"
    # "llama1_13B:paddle_2.5.1:TP2PP4SH1SP1A10040G:1:8:1":"/raid/dataset/llama/"

    # "gpt3_6.7B:paddle_2.5.1:TP1PP1SH2SP8A10040G:1:8:1":"/raid/dataset/gpt-3/"
    # "gpt3_6.7B:paddle_2.5.1:TP2PP1SH1SP4A10040G:1:8:1":"/raid/dataset/gpt-3/"
    # "gpt3_6.7B:paddle_2.5.1:TP2PP1SH2SP4A10040G:1:8:1":"/raid/dataset/gpt-3/"
    # "gpt3_6.7B:paddle_2.5.1:TP2PP4SH1SP1A10040G:1:8:1":"/raid/dataset/gpt-3/"
    # "gpt3_13B:paddle_2.5.1:TP1PP1SH2SP8A10040G:1:8:1":"/raid/dataset/gpt-3/"
    # "gpt3_13B:paddle_2.5.1:TP2PP1SH1SP4A10040G:1:8:1":"/raid/dataset/gpt-3/"
    # "gpt3_13B:paddle_2.5.1:TP2PP1SH2SP4A10040G:1:8:1":"/raid/dataset/gpt-3/"
    # "gpt3_13B:paddle_2.5.1:TP2PP4SH1SP1A10040G:1:8:1":"/raid/dataset/gpt-3/"
    
    # "qwen1.5_MoE:megatron_pai:A800:1:8:1":"/raid/datasets/qwen1.5_MoE/"
    # "mixtral_8x7B:megatron_core060:H100:4:8:1": "/raid/datasets/mistral"

    # kunlunxin cases
    # "gpt2:pytorch:R300:1:8:1": "/raid/dataset/gpt2",
    # "resnet50:pytorch:R300:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "mask_rcnn:pytorch:R300:1:8:1": "/raid/dataset/coco2017/",
    # "retinanet:pytorch:R300:1:8:1": "/raid/dataset/coco2017/",
    # "transformer_xl:pytorch:R300:1:8:1": "/raid/dataset/transformer_xl/",
    # "faster_rcnn:pytorch:R300:1:8:1": "/raid/dataset/coco2017",
    # "transformer_xl:pytorch:R300:1:8:1": "/raid/dataset/transformer_xl/",
    # "glm:pytorch:R300:1:8:1": "/raid/home_datasets_ckpt/glm/train/",
    # "mobilenetv2:pytorch:R300:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "vit:pytorch:R300:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "bert:pytorch:R300:1:8:1": "/raid/dataset/bert_large/train",
    # "longformer:pytorch:R300:1:8:1": "/raid/dataset/longformer_train",
    # "distilbert:pytorch:R300:1:8:1": "/raid/dataset/distilbert/",
    # "swin_transformer:pytorch:R300:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "tacotron2:pytorch:R300:1:8:1": "/raid/dataset/tacotron2/LJSpeech/",
    # "transformer:pytorch:R300:1:8:1": "/raid/dataset/transformer/wmt14_en_de_joined_dict",
    # "bigtransfer:pytorch:R300:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "efficientnet:pytorch:R300:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "llama2_70B:megatron:R300:10:8:1": "/raid/dataset/llama2_70B_pretrain",
    # "baichuan2_13b:deepspeed:R300:1:8:1": "/raid/dataset/baichuan_data/",
    # "baichuan2_13b:deepspeed_new:R300:1:1:1": "/raid/dataset/baichuan_data/",
    # "chatglm3_6b:deepspeed_v0.14.4:R300:1:8:1": "/raid/dataset/chatglm3_6b_data/",

    # iluvatar cases
    # "bigtransfer:pytorch:BI-V100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "vit:pytorch:BI-V100:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "transformer:pytorch:BI-V100:1:8:1": "/raid/dataset/transformer/wmt14_en_de_joined_dict",
    # "bert_hf:pytorch:BI-V100:1:8:1": "/raid/dataset/bert_hf_train",
    # "t5_small:pytorch:BI-V100:1:8:1": "/raid/dataset/t5_small",
    # "baichuan2_13b:deepspeed:BI-V150:2:8:1": "/raid/dataset/baichuan2_13b",
    # "llava1.5_13b:deepspeed-torch:BI-V150:1:16:1": "/raid/dataset/llava1.5_13b",
    # "mixtral_8x7B:megatron:BI-V150:4:16:1": "/raid/dataset/mixtral_8x7B",   ##单机测试
    # "mixtral_8x7B:megatron:BI-V150:1:16:1": "/raid/dataset/mixtral_8x7B",   ##四机测试

    # mthreads cases
    # "resnet50:pytorch_2.0:S4000:1:8:1": "/data/flagperf/ImageNet",
    # "retinanet:pytorch_2.0:S4000:1:8:1": "/data/flagperf/coco2017",
    # "bert_hf:pytorch_2.0:S4000:1:8:1": "/data/flagperf/bert_hf",
    # "llama2_7b:deepspeed:S4000:1:8:1": "/data/flagperf/llama/openwebtext",

    # metax cases
    #"llama3_8B:megatron_core060:C500:1:8:1": "/data/llama3_8b_pretrain"
    # "aquila2_7b:flagscale:C500:1:8:1": "/raid/dataset/Aquila2_7b_data"
    # "faster_rcnn:pytorch_2.0:C500:1:8:1": "/raid/dataset/coco2017/",
    # "retinanet:pytorch_2.0:C500:1:8:1": "/raid/dataset/coco2017/",
    # "resnet50:pytorch_2.0:C500:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "swin_transformer:pytorch_2.0:C500:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "baichuan2_13b:deepspeed:C500:2:8:1": "/raid/dataset/baichuan2_13b",

    # "transformer_xl:pytorch_2.0:C500:1:8:1": "/raid/dataset/transformer_xl/",
    # "wav2vec2:pytorch_2.0:C500:1:8:1": "/raid/dataset/wav2vec2_data/LibriSpeech",
    # "WaveGlow:pytorch_2.0:C500:1:8:1": "/raid/dataset/LJSpeech/",
    # "bert_hf:pytorch_2.0:C500:1:8:1": "/raid/dataset/bert_hf_train",
    # "glm:pytorch_2.0:C500:1:8:1": "/raid/home_datasets_ckpt/glm/train/",
    # "mobilenetv2:pytorch_2.0:C500:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "mask_rcnn:pytorch_2.0:C500:1:8:1": "/raid/dataset/coco2017/",
    # "detr:pytorch_2.0:C500:1:8:1": "/raid/dataset/coco2017/",
    # "transformer:pytorch_2.0:C500:1:8:1": "/raid/dataset/transformer/wmt14_en_de_joined_dict",
    # "cpm:pytorch_2.0:A100:1:8:1": "/raid/dataset/cpm/train/",
    # "distilbert:pytorch_2.0:A100:1:8:1": "/raid/dataset/distilbert/",
    # dcu cases
    # "glm:pytorch_1.13:K100:1:8:1": "/home/chenych/datasets/glm_train_datset/",
    # "bigtransfer:pytorch_2.0:C500:1:8:1": "/raid/dataset/ImageNet_1k_2012/",
    # "longformer:pytorch_2.0:C500:1:8:1": "/raid/dataset/longformer_train/",
    # "llama1_7B:paddle_2.6.0:TP1PP1SH2SP8C50080G:1:8:1":"/raid/dataset/llama/"
    #"gpt3_13B:paddle_2.6.0:TP2PP1SH2SP4C50040G:1:8:1":"/raid/data_set/data-gpt3"
    #"gpt3_13B:paddle_2.6.0:TP1PP1SH2SP8C50080G:1:8:1":"/raid/data_set/data-gpt3"
    "qwen1.5_MoE:megatron_pai:C500:1:8:1":"/raid/datasets/qwen1.5_MoE/"
    
}

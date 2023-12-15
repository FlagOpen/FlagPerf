#!/bin/bash

export PATH=/root/miniconda/envs/python38_torch201_cuda/bin:$PATH

export XDNN_FC_GEMM_DTYPE="float32"
export BKCL_FORCE_SYNC=1
export XPU_FC_AUTOTUNE_FILE="/data/dataset/llama2-7b/fc_autotune_fp16.log"

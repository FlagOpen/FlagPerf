#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

CURPATH="$(dirname "$0")"
# shellcheck source=/dev/null

if [ $# != 3 ] && [ $# != 5 ]; then
    echo "Usage: bash run_standalone_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)"
    echo "Examples:"
    echo "  Train from the beginning:"
    echo "    bash run_standalone_train_gpu.sh /path/to/train.py resnet50_config.yaml /path/to/dataset"
    echo "  Train from full precision checkpoint:"
    echo "    bash run_standalone_train_gpu.sh /path/to/train.py resnet50_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt"
    echo "  Train from pretrained checkpoint:"
    echo "    bash run_standalone_train_gpu.sh /path/to/train.py resnet50_config.yaml /path/to/dataset PRETRAINED /path/to/pretrained_ckpt"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PYTHON_PATH=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
DATASET_PATH=$(get_real_path $3)

if [ ! -d $PYTHON_PATH ]
then
    echo "error: PYTHON_PATH=$PYTHON_PATH is not a directory"
    exit 1
fi

if [ ! -f $CONFIG_FILE ]
then
    echo "error: CONFIG_FILE=$CONFIG_FILE is not a file"
    exit 1
fi

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
    exit 1
fi 

if [ $# == 5 ]; then
  CKPT_TYPE=$4
  CKPT_FILE=$(get_real_path $5)

  if [ "x$CKPT_TYPE" != "xFP32" ] && [ "x$CKPT_TYPE" != "xPRETRAINED" ]; then
      echo "error: CKPT_TYPE=$CKPT_TYPE is not valid, should be FP32 or PRETRAINED"
      exit 1
  fi
  if [ ! -f $CKPT_FILE ]; then
      echo "error: CKPT_FILE=$CKPT_FILE is not a file"
      exit 1
  fi
fi

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ${PYTHON_PATH}/*.py ./train
cp -r ${CURPATH}/../../src ./train
cd ./train || exit

if [ "x$CKPT_TYPE" == "xFP32" ]; then
  python train.py --config_path=$CONFIG_FILE --run_distribute=False --device_target="GPU" \
         --data_path=$DATASET_PATH --fp32_ckpt=$CKPT_FILE --output_path './output' &> log &
elif [ "x$CKPT_TYPE" == "xPRETRAINED" ]; then
  python train.py --config_path=$CONFIG_FILE --run_distribute=False --device_target="GPU" \
         --data_path=$DATASET_PATH --pre_trained=$CKPT_FILE --output_path './output' &> log &
else
  python train.py --config_path=$CONFIG_FILE --run_distribute=False --device_target="GPU" \
         --data_path=$DATASET_PATH --output_path './output' &> log &
fi

#!/bin/bash
set -xe
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install loguru schedule protobuf sentencepiece datasets==2.15.0 schedule==1.2.2 safetensors==0.4.3 numpy==1.26.4
pip uninstall -y transformer-engine
# transformers and accelarate
git clone https://gitee.com/xiaoqi25478/cambricon_wheels.git
cd cambricon_wheels/transformers
pip install -e .
cd ../accelerate
pip install -e .
cd ../../
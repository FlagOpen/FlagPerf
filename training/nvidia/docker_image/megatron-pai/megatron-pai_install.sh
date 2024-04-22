#!/bin/bash
git clone https://github.com/alibaba/Pai-Megatron-Patch.git
cd /workspace/Pai-Megatron-Patch
git checkout aa7c56272cb53a7aeb7fa6ebbfa61c7fa3a5c2e4
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
git submodule init
git submodule update Megatron-LM-240405
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/Pai-Megatron-Patch/Megatron-LM-240405:/workspace/Pai-Megatron-Patch' >> /root/.bashrc
echo 'export EXECPATH=/workspace/Pai-Megatron-Patch/examples/qwen1_5' >> /root/.bashrc
echo 'export CUDA_DEVICE_MAX_CONNECTIONS=1' >> /root/.bashrc
source /root/.bashrc
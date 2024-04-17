#!/bin/bash
# using github mirrors to avoid github TTL
git clone -b core_r0.5.0 https://githubfast.com/NVIDIA/Megatron-LM.git
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM' >> /root/.bashrc
source /root/.bashrc

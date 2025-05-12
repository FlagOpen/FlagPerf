#!/bin/bash
# using github mirrors to avoid github TTL
cp -r /data/dataset/llava/FlagScale /workspace/
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale' >> /root/.bashrc
source /root/.bashrc
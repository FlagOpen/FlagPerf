#!/bin/bash
# using github mirrors to avoid github TTL
cp -r /root/FlagScale /workspace/
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale' >> /root/.bashrc
source /root/.bashrc
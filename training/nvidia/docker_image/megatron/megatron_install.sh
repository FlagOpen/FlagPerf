#!/bin/bash
# using github mirrors to avoid github TTL
git clone -b kunlunxin_llama70B https://github.com/jamesruio/FlagScale.git
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale' >> /root/.bashrc
source /root/.bashrc
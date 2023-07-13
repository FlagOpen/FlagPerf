#!/bin/bash

export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600

if [ -f /usr/local/Ascend/nnae/set_env.sh ];then
    source /usr/local/Ascend/nnae/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f ~/Ascend/nnae/set_env.sh ]; then
    source ~/Ascend/nnae/set_env.sh
elif [ -f ~/Ascend/ascend-toolkit/set_env.sh ]; then
    source ~/Ascend/ascend-toolkit/set_env.sh
else
    echo "warning find no env so not set"
fi

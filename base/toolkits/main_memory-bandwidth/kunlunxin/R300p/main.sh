#!/bin/bash

CASE=$(readlink -f ../.. | awk -F/ '{print $NF}')
pushd /opt/util/examples/$CASE

make clean

export XRE_PATH=/opt/xre
export XDNN_PATH=/opt/xhpc/xdnn
export CXX=g++ 
export XTDK_PATH=/opt/xtdk/ 
export LINK_TYPE=dynamic 

make && ./bandwidth
make clean

popd

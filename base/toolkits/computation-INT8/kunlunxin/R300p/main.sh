#!/bin/bash

CASE=$(readlink -f ../.. | awk -F/ '{print $NF}')
pushd /opt/util/examples/$CASE

make clean

export XRE_PATH=/opt/xre
export XBLAS_PATH=/opt/xhpc/xblas
export CXX=g++ 
export XTDK_PATH=/opt/xtdk/ 
export LINK_TYPE=dynamic 

make && ./gemm 
make clean

popd

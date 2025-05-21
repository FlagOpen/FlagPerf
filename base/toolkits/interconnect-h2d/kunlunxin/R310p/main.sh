#!/bin/bash

TOOL=test_dma
LOG=_${TOOL}.log.${RANDOM}.$$
PERF=/opt/xre/tools/$TOOL
DEV=0
SIZE=$((1024*1024*1024))

numactl --cpunodebind=0 $PERF \
    --loop 5000 \
    $DEV \
    $SIZE | tee $LOG
    
busbw=$(cat ${LOG} | grep -A 4 HOST_TO_DEVICE | tail -1 | cut -d: -f2 | sed -e 's/ //g')
echo "[FlagPerf Result] interconnect-h2d bandwidth=$busbw GB/s"
rm -f $LOG

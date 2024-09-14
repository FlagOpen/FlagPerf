#!/bin/bash

TOOL=all_reduce
LOG=_${TOOL}.log.$$
PERF=/opt/xccl/perf/${TOOL}

$PERF \
    --nxpus 8 \
    --warmup_iters 20 \
    --iters 20000 \
    --minbytes 128m \
    --maxbytes 128m \
    --op_type sum \
    --data_type float \
    -c 0 | tee $LOG

algbw=$(tail -n 1 ${LOG} | awk '{print $6}')
busbw=$(tail -n 1 ${LOG} | awk '{print $NF}')
algbw_bi=$(python3 -c "print(float($algbw) * 2)")
busbw_bi=$(python3 -c "print(float($busbw) * 2)")
echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=${busbw_bi} GB/s"
rm -f $LOG

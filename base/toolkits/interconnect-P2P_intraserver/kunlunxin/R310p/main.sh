#!/bin/bash

source /root/.bashrc
export LD_LIBRARY_PATH=/opt/xre/so:/opt/xccl/so:/opt/mpi/lib/

TOOL=sendrecv
LOG=_${TOOL}.log.${RANDOM}.$$
PERF=/opt/xccl/perf/${TOOL}

$PERF \
    --nxpus 8 \
    --warmup_iters 20 \
    --iters 50000 \
    --minbytes 256m \
    --maxbytes 256m \
    --op_type sum \
    --data_type float \
    -c 0 | tee $LOG

algbw=$(tail -n 1 ${LOG} | awk '{print $6}')
busbw=$(tail -n 1 ${LOG} | awk '{print $NF}')
algbw_bi=$(python3 -c "print(float($algbw) * 2)")
busbw_bi=$(python3 -c "print(float($busbw) * 2)")
echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=${busbw_bi} GB/s"
rm -f $LOG

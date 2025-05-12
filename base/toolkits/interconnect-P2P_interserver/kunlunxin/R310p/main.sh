#!/bin/bash

set -x

/etc/init.d/ssh start
/etc/init.d/ssh status
sleep 3
source /root/.bashrc
export LD_LIBRARY_PATH=/opt/xre/so:/opt/xccl/so:/opt/mpi/lib/

#hosts
hosts=$(cat "../../../../configs/host.yaml" | egrep -v '^\s*#' | grep HOSTS: | cut -d: -f2| perl -F, -lne '/(\d+\.\d+\.\d+\.\d+)/ && push @h,$1.":1" foreach @F; print join(",", @h)')
n=$(($(echo $hosts| sed -e 's/,/\n/g'| wc -l)*8))

TOOL=sendrecv
LOG=_${TOOL}.log.${RANDOM}.$$
PERF=/opt/xccl/perf/${TOOL}

if [[ w"$NODERANK" != w"0" ]]; then
    echo "launch mpirun only on first node, exiting.\n"
    exit
fi

mpirun -hosts "${hosts}" -n 2 $PERF \
    --nxpus 2\
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
echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=${busbw} GB/s"
rm -f $LOG
rm -f $HOSTFILE

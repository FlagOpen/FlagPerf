LOG_PATH=$(pwd)/$(ip a | grep -w 'inet' | grep 'global' | sed 's/.*inet //;s/\/.*//' | awk 'NR==1{print $1}')_run_log

mpirun -x NCCL_GRAPH_FILE=/opt/single_graph.xml -x NCCL_P2P_LEVEL=SYS -x NCCL_MIN_NCHANNELS=32 -x NCCL_MIN_P2P_NCHANNELS=32 -x RCCL_SDMA_COUNT_ENABLE=1 -x RCCL_SDMA_COPY_ENABLE=1 -x RCCL_COLL_XHCL_CHANNEL_NUM=28 --allow-run-as-root -np 8 all_reduce_perf -b 1g -e 1g -f 2 -g 1 2>&1 | tee ${LOG_PATH}

data=$(grep "# Avg bus bandwidth" ${LOG_PATH} | awk '{print $NF}')
result=$(python3 -c "print(float($data) * 2)")
echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=$result GB/s"
rm -rf ${LOG_PATH}
LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
mpirun -x NCCL_GRAPH_FILE=/opt/single_graph.xml -x NCCL_P2P_LEVEL=SYS -x NCCL_MIN_NCHANNELS=32 -x NCCL_MIN_P2P_NCHANNELS=32 --allow-run-as-root -np 8 all_reduce_perf -b 256M -e 256M -f 2 -g 1 2>&1 | tee ${LOG_PATH}

data=$(grep "# Avg bus bandwidth" ${LOG_PATH} | awk '{print $NF}')

echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=$data GB/s"
rm -rf ${LOG_PATH}
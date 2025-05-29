#!/bin/bash

curr_path=$(pwd)

vendor_path=../../../../vendors/tsingmicro
exec_path=/root/TX81
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$exec_path


cd $exec_path
./tsmPerf -r allreducePerf --multi-node | tee /tmp/multi_mpi_test_result.log
cd -

python3 $vendor_path/log_analysis.py --log_type="inter_allreduce" --log_file="/tmp/multi_mpi_test_result.log"



#!/bin/bash

curr_path=$(pwd)

cd ../../../../vendors/tsingmicro/TX81/
./tsmPerf -r allreducePerf --multi-node | tee $curr_path/allreduce_result.log
cd -

python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="inter_allreduce" --log_file="./allreduce_result.log"
rm -f ./allreduce_result.log

#!/bin/bash

curr_path=$(pwd)

vendor_path=../../../../vendors/tsingmicro
exec_path=/root/TX81
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$exec_path

cd $exec_path
./tsmPerf -r tanhPerf -c $curr_path/tsm.example.yml -d chip_out/ -f 0 | tee $curr_path/computation_result.log
cd -

python3 $vendor_path/log_analysis.py --log_type="computation_fp32" --log_file="./computation_result.log"
rm -f ./computation_result.log

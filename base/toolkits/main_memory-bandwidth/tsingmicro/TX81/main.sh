#!/bin/bash

# tsmvs -c ddr_perf_test.yaml 2>&1 | tee ./ddr_test.log
# cat ./ddr_test.log | grep ddr_bandwidth > ./ddr_bandwidth.log
# python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="ddr_perf" --log_file="./ddr_bandwidth.log"
# rm -f ./ddr_test.log
# rm -f ./ddr_bandwidth.log

curr_path=$(pwd)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$curr_path/../../../../vendors/tsingmicro/TX81/

cd ../../../../vendors/tsingmicro/TX81/
chmod +x tsmPerf
./tsmPerf -r lsuPerf -d chip_out/ -f 0 | tee $curr_path/test_result.log
cd -

python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="lsu_test" --log_file="./test_result.log"


#!/bin/bash

curr_path=$(pwd)

vendor_path=../../../../vendors/tsingmicro
exec_path=/root/TX81
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$exec_path

cd $exec_path
./tsmPerf -r lsuPerf -d chip_out/ -f 0 | tee $curr_path/test_result.log
cd -

python3 $vendor_path/log_analysis.py --log_type="lsu_test" --log_file="./test_result.log"
rm -f ./test_result.log


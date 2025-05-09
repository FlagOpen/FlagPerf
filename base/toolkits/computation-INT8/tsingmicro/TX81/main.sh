#!/bin/bash

curr_path=$(pwd)

cd ../../../../vendors/tsingmicro/TX81/
bash main.sh $curr_path/tsm.example.yml | tee $curr_path/computation_result.log
cd -

python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="computation_int8" --log_file="./computation_result.log"
rm -f ./computation_result.log



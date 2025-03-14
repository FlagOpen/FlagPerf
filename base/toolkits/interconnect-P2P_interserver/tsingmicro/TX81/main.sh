#!/bin/bash

tsmvs -c c2c_global_perf_test.yaml 2>&1 | tee ./c2c_test.log

python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="c2c_global_perf" --log_file="./c2c_test.log"
rm -f ./c2c_test.log

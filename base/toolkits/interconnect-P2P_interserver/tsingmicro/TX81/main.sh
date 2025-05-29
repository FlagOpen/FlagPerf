#!/bin/bash

tsmvs -c c2c_global_perf_test.yaml 2>&1 | tee /tmp/c2c_global_test.log

python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="c2c_global_perf" --log_file="/tmp/c2c_global_test.log"



#!/bin/bash

tsmvs -c ddr_perf_test.yaml | tee ./ddr_test.log
cat ./ddr_test.log | grep ddr_bandwidth > ./ddr_bandwidth.log
# python3 log_analysis.py --log_type="ddr_perf" --log_file="./ddr_bandwidth.log"
python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="ddr_perf" --log_file="./ddr_bandwidth.log"
rm -f ./ddr_test.log
rm -f ./ddr_bandwidth.log

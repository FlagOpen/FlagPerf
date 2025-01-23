#!/bin/bash

tsmvs -c npu_stress_test.yaml | tee ./power_test.log
# python3 ../../../main_memory-bandwidth/tsingmicro/TX81/log_analysis.py --log_type="power_full_load" --log_file="./power_test.log"
python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="power_full_load" --log_file="./power_test.log"
rm -f ./power_test.log

#!/bin/bash

tsm_smi | tee ./tsm_smi.log
cat ./tsm_smi.log | grep TX81 > ./ddr_capacity.log
# python3 ../../../main_memory-bandwidth/tsingmicro/TX81/log_analysis.py --log_type="ddr_cap" --log_file="./ddr_capacity.log"
python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="ddr_cap" --log_file="./ddr_capacity.log"
rm -f ./tsm_smi.log
rm -f ./ddr_capacity.log


#!/bin/bash

tsmvs -c pcie_perf_test.yaml 2>&1 | tee ./pcie_test.log

# python3 ../../../main_memory-bandwidth/tsingmicro/TX81/log_analysis.py --log_type="pcie_perf" --log_file="./pcie_test.log"
python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="pcie_perf" --log_file="./pcie_test.log"
rm -f ./pcie_test.log


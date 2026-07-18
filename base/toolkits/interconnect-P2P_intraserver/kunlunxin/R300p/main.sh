#!/bin/bash


# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TOOL=sendrecv
LOG=_${TOOL}.log.${RANDOM}.$$
PERF=/opt/xccl/perf/${TOOL}

$PERF \
    --nxpus 8 \
    --warmup_iters 20 \
    --iters 50000 \
    --minbytes 256m \
    --maxbytes 256m \
    --op_type sum \
    --data_type float \
    -c 0 | tee $LOG

algbw=$(tail -n 1 ${LOG} | awk '{print $6}')
busbw=$(tail -n 1 ${LOG} | awk '{print $NF}')
algbw_bi=$(python3 -c "print(float($algbw) * 2)")
busbw_bi=$(python3 -c "print(float($busbw) * 2)")
echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=${busbw_bi} GB/s"
rm -f $LOG

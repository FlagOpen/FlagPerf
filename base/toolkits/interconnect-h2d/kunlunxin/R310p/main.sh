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

TOOL=test_dma
LOG=_${TOOL}.log.${RANDOM}.$$
PERF=/opt/xre/tools/$TOOL
DEV=0
SIZE=$((1024*1024*1024))

numactl --cpunodebind=0 $PERF \
    --loop 5000 \
    $DEV \
    $SIZE | tee $LOG
    
busbw=$(cat ${LOG} | grep -A 4 HOST_TO_DEVICE | tail -1 | cut -d: -f2 | sed -e 's/ //g')
echo "[FlagPerf Result] interconnect-h2d bandwidth=$busbw GB/s"
rm -f $LOG

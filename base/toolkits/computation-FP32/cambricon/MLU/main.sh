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

export MLU_VISIBLE_DEVICES=0
LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
cnvs -r matmul_performance -c `pwd`/cnvs.example.yml 2>&1 | tee ${LOG_PATH}
value=$(grep -o 'matmul performance(GOPS): [0-9.]\+' ${LOG_PATH} )
number=$(echo $value | grep -o '[0-9.]\+')
result=$(python3 -c "print(float($number) / 1000)")
echo "[FlagPerf Result] computation-FP32=$result TFLOPS"
rm -rf cnvs_stats ${LOG_PATH} #删除缓存文件

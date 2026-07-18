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

source /usr/local/Ascend/toolbox/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
LOG_PATH=`pwd`
npu-smi info -t memory -i 0 -c 0 > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(grep "HBM Capacity(MB)" ${LOG_PATH}/test_result.log | awk '{print $NF}')
RESULT_A3=$(expr $RESULT \* 2)
echo "[FlagPerf Result] main_memory-capacity=${RESULT_A3} MiB"
rm -rf ${LOG_PATH}/test_result.log

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
ascend-dmi --bw -t p2p --sp 0 -q --ip 127.0.0.1 --spp /home/zhiyuan/share_path --hip 127.0.0.2 -m card > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(awk 'NR>13 && NR<15 {print $3}' ${LOG_PATH}/test_result.log)
echo "[FlagPerf Result] interconnect-P2P_interserver-bandwidth=${RESULT} GB/s"
rm -rf ${LOG_PATH}/test_result.log

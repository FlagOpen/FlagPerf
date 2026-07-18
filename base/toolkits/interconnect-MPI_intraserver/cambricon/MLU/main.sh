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

LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
/usr/local/neuware/bin/allreduce \
    --warmup_loop 20  \
    --thread 8 \
    --loop 2000 \
    --mincount 1 \
    --maxcount 512M \
    --multifactor 2 \
    --async 1 \
    --block 0 2>&1 | tee ${LOG_PATH}
data=$(tail -n 2 ${LOG_PATH} | awk '{print $11 }')
result=$(python3 -c "print(float($data) * 2)")
echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=$result GB/s"
rm -rf ${LOG_PATH} #删除缓存文件

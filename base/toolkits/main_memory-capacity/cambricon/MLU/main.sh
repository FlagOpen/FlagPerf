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
pushd /usr/local/neuware/samples/cnrt && mkdir -p build && pushd build && cmake .. && make -j20 && pushd bin 
for i in $(seq 1 7500)  
do
    echo ${i}
    ./basic_device_info 2>&1 | tee ${LOG_PATH}
done
value=$(grep "Device 0 has avaliable memory in MB" "$LOG_PATH" | awk '{print $8}')
echo "[FlagPerf Result]main_memory-capacity=${value} MiB"
rm -rf ${LOG_PATH} #删除缓存文件
popd && popd && popd

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

LOG_PATH=$(pwd)/$(ip a | grep -w 'inet' | grep 'global' | sed 's/.*inet //;s/\/.*//' | awk 'NR==1{print $1}')_run_log

export PATH=/opt/hyqual_v3.0.3:${PATH}
run 7 2>&1 | tee ${LOG_PATH}

data=$(grep 'peak dgemm    :'  ${LOG_PATH} | awk '{print $4}' | sort -nr | head -n1)
echo "[FlagPerf Result]computation-FP64=$data TFLOPS"
rm -rf ${LOG_PATH}

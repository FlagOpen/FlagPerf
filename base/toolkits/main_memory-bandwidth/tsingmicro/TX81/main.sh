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


#!/bin/bash


source /root/.bash_profile
tsmvs -c ddr_perf_test.yaml 2>&1 | tee ./ddr_test.log
cat ./ddr_test.log | grep ddr_bandwidth > ./ddr_bandwidth.log
python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="ddr_perf" --log_file="./ddr_bandwidth.log"
rm -f ./ddr_test.log
rm -f ./ddr_bandwidth.log

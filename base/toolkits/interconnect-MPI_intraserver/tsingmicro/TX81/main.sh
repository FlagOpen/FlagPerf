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

curr_path=$(pwd)

vendor_path=../../../../vendors/tsingmicro
exec_path=/root/TX81
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$exec_path

cd $exec_path
./tsmPerf -r allreducePerf -i 10 -f 0 | tee $curr_path/test_result.log
cd -

python3 $vendor_path/log_analysis.py --log_type="intra_allreduce" --log_file="./test_result.log"
rm -f ./test_result.log

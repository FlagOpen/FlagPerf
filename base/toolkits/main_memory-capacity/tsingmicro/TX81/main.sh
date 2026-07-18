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

tsm_smi | tee ./tsm_smi.log
cat ./tsm_smi.log | grep TX81 > ./ddr_capacity.log
# python3 ../../../main_memory-bandwidth/tsingmicro/TX81/log_analysis.py --log_type="ddr_cap" --log_file="./ddr_capacity.log"
python3 ../../../../vendors/tsingmicro/log_analysis.py --log_type="ddr_cap" --log_file="./ddr_capacity.log"
rm -f ./tsm_smi.log
rm -f ./ddr_capacity.log

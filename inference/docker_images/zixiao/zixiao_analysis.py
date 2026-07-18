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

def analysis_log(logpath):
    logfile = open(logpath)

    max_usage = 0.0 ## usage_mem
    max_mem = 16.0 
    for line in logfile.readlines():
        '''
        zx-smi pwr DTemp MUsed Mem
        '''
        if "zx-smi" in line:
            line = line[:-1]
            usage = line.split(" ")[3]
            usage = float(usage)*16/100
            max_usage = max(max_usage, usage)
    return round(max_usage, 2), max_mem, eval("30e12"), eval("120e12")

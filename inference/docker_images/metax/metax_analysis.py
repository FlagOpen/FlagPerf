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

    max_usage = 0.0
    max_mem = 0.0
    for line in logfile.readlines():
        if "MiB" in line:

            usage_and_maxusage = line.split(" ")[2]
            usage = float(usage_and_maxusage.split("/")[0])
            max_usage = max(max_usage, usage)
            max_mem = float(usage_and_maxusage.split("/")[1])
            #max_mem = float(max_mem[:-3])
            print (max_mem)
            print (max_usage)
    return round(max_usage / 1024.0,
                 2), round(max_mem / 1024.0, 2), eval("120e12"), eval("240e12")

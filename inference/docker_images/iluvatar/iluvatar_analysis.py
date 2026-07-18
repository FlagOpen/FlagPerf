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
            usage = line.split(" ")[2]
            usage = float(usage[:-3])
            max_usage = max(max_usage, usage)
            if (max_usage >= usage):
                m_mem = line.split(" ")[3]
                m_mem = float(m_mem[:-3])
                max_mem = max(max_mem, m_mem)

    return round(max_usage / 1024.0,
                 2), round(max_mem / 1024.0, 2), eval("24e12"), eval("96e12")

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

def analysis_log(logpath, config):
    logfile = open(logpath)

    result = {"temp": {}, "power": {}, "mem": {}}
    for gpuID in range(config.NPROC_PER_NODE):
        for monitor_index in result.keys():
            result[monitor_index][gpuID] = []

    max_mem = None
    next_gpu_id = 0
    for line in logfile.readlines():
        #35C 58W 280W 666/65536 MiB
        if "MiB" in line:

            if max_mem is None:
                usage_and_maxusage = line.split(" ")[2]
                result["max_mem"] = float(usage_and_maxusage.split("/")[1])
            temp_str = line.split(" ")[0]
            temp =  (float(temp_str[:-1]))
            power_str = line.split(" ")[1]
            power =  (float(power_str[:-1]))
            usage_and_maxusage = line.split(" ")[2]
            usage = float(usage_and_maxusage.split("/")[0])
            max_mem = float(usage_and_maxusage.split("/")[1])
            result["temp"][next_gpu_id].append(temp)
            result["power"][next_gpu_id].append(power)
            result["mem"][next_gpu_id].append(usage)
            next_gpu_id = (next_gpu_id + 1) % config.NPROC_PER_NODE

    return result

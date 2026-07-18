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
        if "MiB" in line:
            if max_mem is None:
                max_mem = float(line.split(" ")[3][:-3])
                result["max_mem"] = max_mem
            temp = float(line.split(" ")[0][:-1])
            power = float(line.split(" ")[1][:-1])
            mem = float(line.split(" ")[2][:-3])
            result["temp"][next_gpu_id].append(temp)
            result["power"][next_gpu_id].append(power)
            result["mem"][next_gpu_id].append(mem)
            next_gpu_id = (next_gpu_id + 1) % config.NPROC_PER_NODE

    return result

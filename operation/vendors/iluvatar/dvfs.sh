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

flag=$1

if [[ $flag != 0 ]] && [[ $flag != 1 ]]; then
    echo "Wrong target flag: $flag"
    exit
fi

# a=$(ixsmi -q|grep 'Bus Id'|awk '{print $NF}');
a=$(lspci|grep 1e3e|awk '{print $1}')
for i in ${a[@]};
do
    # bus_id=${i/0000/};
    bus_id=0000:${i}
    # echo "---before set---"
    # cat /sys/bus/pci/devices/${bus_id,,}/itr_debug
    cmd="echo perf_mode $flag > /sys/bus/pci/devices/${bus_id,,}/itr_debug"
    echo $cmd
    eval $cmd
    if [[ $flag == 0 ]]; then
        printf "Turn off DVFS mode: "
    else
        printf "Turn on DVFS mode: "
    fi
    if [[ $? == 0 ]]; then
        echo "Success"
    else
        echo "Failed"
    fi
    # echo "---after  set---"
    # cat /sys/bus/pci/devices/${bus_id,,}/itr_debug
done

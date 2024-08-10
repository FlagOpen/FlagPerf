#!/bin/bash

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

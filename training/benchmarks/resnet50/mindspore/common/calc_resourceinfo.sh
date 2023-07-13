#!/bin/bash
export npu_info=/var/log/npu_info.log
export gpu_info=/var/log/gpu_info.log
npu_monitor_pid=0
gpu_monitor_pid=0

function calc_resourceinfo_npu()
{
  device_group_xp="$*"
  npuandchip=($(npu-smi  info -m | egrep -v "ID" | awk '{print $1" "$2" "$3}'| xargs))
  num_i=0
  one_average_usage=0
  set_result_file=$2
  while [[ num_i -lt ${#npuandchip[@]} ]]; do
    for i in ${device_group_xp[*]}; do
      if [ ${i} == ${npuandchip[num_i+2]} ];then
        echo $i | grep -q '[^0-9]'
        nl=$?
        if [ $nl -eq 0 ];then
          continue
        fi
        one_date_usage=$(cat ${npu_info} | awk -v a=${npuandchip[num_i]} -v b=${npuandchip[num_i+1]} '$1==a && $2==b {print $5}')
        onesub_usage=$(echo $one_date_usage |xargs |sed 's/[[:space:]]/\+/g'|bc)
        onesub_num=$(echo $one_date_usage |awk '{print NF}')
        if [ ${onesub_num} -eq 0 ];then
          one_average_usage=0
        else
          one_average_usage=$(awk -v x=${onesub_usage} -v y=${onesub_num} 'BEGIN{print x/y}')
        fi
        echo "resource_util_ratio: $one_average_usage"
        python3 ${set_result_file} "training" "resource_util_ratio" ${one_average_usage}
      fi
    done
    let num_i=num_i+3
  done
}

function calc_runing_resourceinfo_npu() {
  if [ "$npu_monitor_pid" != "" ];then
    kill $npu_monitor_pid > /dev/null 2>&1
  fi
  calc_resourceinfo_npu "average_usage" $*
  rm -rf ${npu_info}
}

function run_resourceinfo_monitor_backgroud_npu() {
  stdbuf -oL npu-smi info watch -d 5 >> ${npu_info} &
  export npu_monitor_pid=$!
}

function calc_resourceinfo_gpu()
{
  device_group_xp="$*"
  one_average_usage=0
  set_result_file=$2
  echo "device_group_xp : " ${device_group_xp}
  for i in ${device_group_xp[*]}; do
      echo $i | grep -q '[^0-9]'
      nl=$?
      if [ $nl -eq 0 ];then
        continue
      fi
      one_date_usage=$(cat ${gpu_info} | awk -v a="${i},"  '$1==a  {print $2}')
      onesub_usage=$(echo $one_date_usage |xargs |sed 's/[[:space:]]/\+/g'|bc)
      onesub_num=$(echo $one_date_usage |awk '{print NF}')
      if [ ${onesub_num} -eq 0 ];then
        one_average_usage=0
      else
        one_average_usage=$(awk -v x=${onesub_usage} -v y=${onesub_num} 'BEGIN{print x/y}')
      fi
      echo "resource_util_ratio: $one_average_usage"
      python3 ${set_result_file} "training" "resource_util_ratio" ${one_average_usage}
  done
}

function calc_runing_resourceinfo_gpu()
{
  if [ "$gpu_monitor_pid" != "" ];then
    kill $gpu_monitor_pid > /dev/null 2>&1
  fi
  calc_resourceinfo_gpu "average_usage" $*
  rm -rf ${gpu_info}
}

function run_resourceinfo_monitor_backgroud_gpu() {
  stdbuf -oL nvidia-smi --query-gpu=index,utilization.gpu --format=csv -l 5 >> ${gpu_info} &
  export gpu_monitor_pid=$!
}

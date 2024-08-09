#!/bin/bash

# ==========================修改点1: START==========================
VENDOR="nvidia"
ACCE_CONTAINER_OPT=" --gpus all"
ACCE_VISIBLE_DEVICE_ENV_NAME="CUDA_VISIBLE_DEVICES"
SSH_PORT="22"
HOSTS_PORTS="[\"2222\"]"
MASTER_PORT="29501"
TDP="400W"

ip_address="10.1.2.155"
chip_name="A100_40_SXM"
env_name="ngctorch2403"

declare -A spec_tflops_dict
spec_tflops_dict["BF16"]=312
spec_tflops_dict["FP16"]=312
spec_tflops_dict["FP32"]=19.5
#=============================STOP==========================

declare -A op_dict
# Assign list values to dictionary keys
# ==========================修改点2: START==========================
op_dict["sum"]="FP32 FP16 BF16"
op_dict["sub"]="FP32 FP16 BF16"
op_dict["mm"]="FP32 FP16 BF16"
op_dict["abs"]="FP32 FP16 BF16"
op_dict["add"]="FP32 FP16 BF16"
op_dict["all"]="FP32 FP16 BF16"
#op_dict["bitwise_and"]="INT32 INT16"
#op_dict["bitwise_not"]="INT32 INT16"
#op_dict["bitwise_or"]="INT32 INT16"
op_dict["bmm"]="FP32 FP16 BF16"
op_dict["cos"]="FP32 FP16 BF16"
op_dict["div"]="FP32 FP16 BF16"
op_dict["eq"]="FP32 FP16 BF16"
op_dict["exp"]="FP32 FP16 BF16"
op_dict["ge"]="FP32 FP16 BF16"
op_dict["gelu"]="FP32 FP16 BF16"
op_dict["gt"]="FP32 FP16 BF16"
op_dict["isinf"]="FP32 FP16 BF16"
op_dict["isnan"]="FP32 FP16 BF16"
op_dict["le"]="FP32 FP16 BF16"
#op_dict["linear"]="FP32 FP16 BF16"
op_dict["lt"]="FP32 FP16 BF16"
op_dict["max"]="FP32 FP16 BF16"
op_dict["mean"]="FP32 FP16 BF16"
op_dict["min"]="FP32 FP16 BF16"
op_dict["mul"]="FP32 FP16 BF16"
op_dict["mv"]="FP32 FP16 BF16"
op_dict["ne"]="FP32 FP16 BF16"
op_dict["neg"]="FP32 FP16 BF16"
#op_dict["or"]="FP32 FP16 BF16"
op_dict["pow"]="FP32 FP16 BF16"
op_dict["prod"]="FP32 FP16 BF16"
op_dict["reciprocal"]="FP32 FP16 BF16"
op_dict["rsqrt"]="FP32 FP16 BF16"
op_dict["sin"]="FP32 FP16 BF16"
#=============================STOP==========================


export VENDOR
export ACCE_CONTAINER_OPT
export ACCE_VISIBLE_DEVICE_ENV_NAME
export SSH_PORT
export HOSTS_PORTS
export MASTER_PORT
export TDP

file="overall.data"

# 检查文件是否存在
if [ -e "$file" ]; then
  rm "$file"
  echo "$file 已被删除"
fi
touch "$file"

total=0
success=0
fail=0

# Iterate over op_dict
for key in "${!op_dict[@]}"; do
    IFS=' ' read -r -a value_list <<< "${op_dict[$key]}"
    # Iterate over values for each key
    for value in "${value_list[@]}"; do
        # Your code here
        echo "Running operation: $key with data format: $value"
        total=$((total + 1))
        # Example command using the variables
        bash run.sh --op_name "$key" --data_format "$value" --ip_address "$ip_address" --chip_name "$chip_name" --env_name "$env_name" --spec_tflops "${spec_tflops_dict[$value]}"
        if [ $? -eq 0 ]; then
            echo "success: ${key} ${value}" >> $file
            success=$((success + 1))
        else
            echo "fail: ${key} ${value}" >> $file
            fail=$((fail + 1))
        fi
    done
done

echo -e "\n\n\ntotal: ${total}" >> $file
echo "success: ${success}" >> $file
echo "fail: ${fail}" >> $file

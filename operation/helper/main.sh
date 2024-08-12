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
spec_tflops_dict["INT32"]=19.5
spec_tflops_dict["INT16"]=-1
#=============================STOP==========================

declare -A op_dict
# Assign list values to dictionary keys
# ==========================修改点2: START==========================
opdict["abs"]="FP32 FP16 BF16"
opdict["add"]="FP32 FP16 BF16"
opdict["addmm"]="FP32 FP16 BF16"
opdict["all"]="FP32 FP16 BF16"
opdict["amax"]="FP32 FP16 BF16"
opdict["argmax"]="FP32 FP16 BF16"
opdict["bitwise_and"]="INT32 INT16"
opdict["bitwise_not"]="INT32 INT16"
opdict["bitwise_or"]="INT32 INT16"
opdict["bmm"]="FP32 FP16 BF16"
opdict["cos"]="FP32 FP16 BF16"
opdict["cross_entropy_loss"]="FP32 FP16 BF16"
opdict["div"]="FP32 FP16 BF16"
opdict["dropout"]="FP32 FP16 BF16"
opdict["eq"]="FP32 FP16 BF16"
opdict["exp"]="FP32 FP16 BF16"
opdict["ge"]="FP32 FP16 BF16"
opdict["gelu"]="FP32 FP16 BF16"
opdict["group_norm"]="FP32 FP16 BF16"
opdict["gt"]="FP32 FP16 BF16"
opdict["isinf"]="FP32 FP16 BF16"
opdict["isnan"]="FP32 FP16 BF16"
opdict["layer_norm"]="FP32 FP16 BF16"
opdict["le"]="FP32 FP16 BF16"
opdict["linear"]="FP32 FP16 BF16"
opdict["log_softmax"]="FP32 FP16 BF16"
opdict["lt"]="FP32 FP16 BF16"
opdict["max"]="FP32 FP16 BF16"
opdict["mean"]="FP32 FP16 BF16"
opdict["min"]="FP32 FP16 BF16"
opdict["mm"]="FP32 FP16 BF16"
opdict["mul"]="FP32 FP16 BF16"
opdict["mv"]="FP32 FP16 BF16"
opdict["native_dropout"]="FP32 FP16 BF16"
opdict["native_group_norm"]="FP32 FP16 BF16"
opdict["ne"]="FP32 FP16 BF16"
opdict["neg"]="FP32 FP16 BF16"
opdict["pow"]="FP32 FP16 BF16"
opdict["prod"]="FP32 FP16 BF16"
opdict["reciprocal"]="FP32 FP16 BF16"
opdict["relu"]="FP32 FP16 BF16"
opdict["rsqrt"]="FP32 FP16 BF16"
opdict["sigmoid"]="FP32 FP16 BF16"
opdict["silu"]="FP32 FP16 BF16"
opdict["sin"]="FP32 FP16 BF16"
opdict["softmax"]="FP32 FP16 BF16"
opdict["sub"]="FP32 FP16 BF16"
opdict["sum"]="FP32 FP16 BF16"
opdict["tanh"]="FP32 FP16 BF16"
opdict["triu"]="FP32 FP16 BF16"
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

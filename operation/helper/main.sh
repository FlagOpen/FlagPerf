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
#=============================STOP==========================

declare -A op_dict
# Assign list values to dictionary keys
# ==========================修改点2: START==========================
op_dict["sum"]="FP32 FP16"
op_dict["bitwise_or"]="INT8"
#=============================STOP==========================


export VENDOR
export ACCE_CONTAINER_OPT
export ACCE_VISIBLE_DEVICE_ENV_NAME
export SSH_PORT
export HOSTS_PORTS
export MASTER_PORT
export TDP


# Iterate over op_dict
for key in "${!op_dict[@]}"; do
    IFS=' ' read -r -a value_list <<< "${op_dict[$key]}"
    # Iterate over values for each key
    for value in "${value_list[@]}"; do
        # Your code here
        echo "Running operation: $key with data format: $value"
        # Example command using the variables
        bash run.sh --op_name "$key" --data_format "$value" --ip_address "$ip_address" --chip_name "$chip_name" --env_name "$env_name"
    done
done
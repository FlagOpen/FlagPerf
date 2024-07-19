#!/bin/bash

# 函数: 修改配置
modify_config() {
    echo "Modifying configuration..."
    local case_type=$1
    local result_dir=$2
    echo "case_type is: ${case_type}"
    echo "result_dir is: ${result_dir}"
    # 修改 operation/configs/host.yaml
    echo "operationdir is: ${OPERATIONDIR}"
    sed -i "s|^FLAGPERF_PATH:.*$|FLAGPERF_PATH: \"$OPERATIONDIR\"|" "${OPERATIONDIR}/configs/host.yaml"

    # 修改ip
    echo "ip address is: ${ip_address}"
    sed -i "s|^HOSTS:.*$|HOSTS: [\"$ip_address\"]|" "${OPERATIONDIR}/configs/host.yaml"

    # 替换FP  和 替换case 类型等
    #  "eq:FP32:nativetorch:A100_40_SXM": "ngctorch2403"
    sed -i "s|^    \".*:.*:.*:.*\": \".*\"|    \"$op_name:$data_format:$case_type:$chip_name\": \"$env_name\"|" "${OPERATIONDIR}/configs/host.yaml"
    # 备份一下, 方便排查问题
    cp "${OPERATIONDIR}/configs/host.yaml" "${result_dir}/bak_${case_type}_host.yaml"
   }


parse_log() {
    local result_dir=$1
    log_dir="${OPERATIONDIR}/result"
    latest_folder=$(ls -td "$log_dir"/*/ | head -n 1)
    echo "log dir is: ${latest_folder}"
    log_file_path="${latest_folder}flagperf_run.log"
    readme_file_path="${result_dir}"
    if [ -f "$log_file_path" ]; then
        cd "${CURRENTDIR}"
        python render.py "${log_file_path}" "${case_type}" "${readme_file_path}"
    else
        echo "error: log dir not exist"
        exit 1
    fi
}

run_cases_and_gen_readme() {
    local case_type=$1
    local result_dir=$2
    # 执行测试
    cd "$OPERATIONDIR"
    echo "-------------------current dir---------------------"
    echo `pwd`
    echo "start to run..."
    python run.py
    # 检查上一条命令的执行结果
    if [ $? -eq 0 ]; then
        echo "执行成功"
        parse_log $result_dir
    else
        echo "执行失败"
        exit 1
    fi

}



main() {
    result_dir="results/${op_name}_${data_format}"
    mkdir $result_dir
    if [ -f "${result_dir}/data.json" ]; then
        rm "${result_dir}/data.json"
    fi
    # 调用修改配置函数
    case_type=("flaggems" "nativetorch")
    for case_type in ${case_type[@]}
    do
        modify_config "$case_type" "$result_dir"
        run_cases_and_gen_readme "$case_type" "$result_dir"
    done
}

# Initialize variables with default values
data_format=""
op_name=""
ip_address=""
chip_name=""
env_name=""
# Function to display usage information
usage() {
    echo "Usage: $0 --op_name <op_name> --data_format <data_format> --ip_address <ip_address>  --chip_name <chip_name> --env_name <env_name>"
    exit 1
}


# Parse command line options
while [ $# -gt 0 ]; do
    case "$1" in
        --data_format)
            data_format="$2"
            shift 2
            ;;
        --op_name)
            op_name="$2"
            shift 2
            ;;
        --ip_address)
            ip_address="$2"
            shift 2
            ;;
        --chip_name)
            chip_name="$2"
            shift 2
            ;;
        --env_name)
            env_name="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if required options are provided
if [ -z "$data_format" ] || [ -z "$op_name" ] || [ -z "$ip_address" ] || [ -z "$chip_name" ] || [ -z "$env_name" ]; then
    echo "Error: Missing required options."
    usage
fi

# Display parsed options
echo "data_format: $data_format"
echo "op_name: $op_name"
echo "ip_address: $ip_address"
echo "chip_name: $chip_name"
echo "env_name: $env_name"


CURRENTDIR=$(pwd)
# 获取当前路径的上一级目录
OPERATIONDIR=$(dirname "$CURRENTDIR")

main

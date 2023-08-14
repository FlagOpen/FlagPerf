#!/usr/bin/env bash
set +ex

# scripts for distributed training:
# 0. make sure to correctly config ssh connection for <username>, and <username> has sudo privilege.
# 1. make sure configs are correctly configured.(cluster_conf, test_conf)
# 2. distributed training is started on the master server.

# 需要填写的信息
ip_list=("x.x.x.x")        # 填写多机训练的ip列表，不包含 master-server
username="<YOUR_USERNAME>" # 当前用户
dataset_dir=$1             # 数据集路径

# FlagPerf的父级目录
source_directory=$(cd "$(dirname "$0")" && cd ./../../.. && pwd)
echo "source_directory: ${source_directory}"

app_name="FlagPerf"
zip_filename="${app_name}.zip"

timestamp=$(date "+%y%m%d_%H%M%S")
echo "timestamp: ${timestamp}"
echo "datset_dir: ${dataset_dir}"
sleep 2s

if echo "$username" | grep -qi "YOUR_USERNAME"; then
    echo "Please replace username with you real username"
    exit 255
fi

local_zip=$source_directory/$zip_filename
cd $source_directory && rm -fv $local_zip
cd $source_directory && zip -r $zip_filename $app_name

for ip in "${ip_list[@]}"; do

    if [ $ip == "x.x.x.x" ]; then
        echo "Please replace ip_list to real ip list for your servers"
        exit 255
    fi

    if [ -n "$dataset_dir" ]; then
        dataset_parent_dir=$(dirname $dataset_dir)
        echo "dataset_parent_dir: $dataset_parent_dir"

        # 在远程服务器上创建目录
        ssh $username@$ip "sudo mkdir -p $dataset_parent_dir"
        echo "mkdir $dataset_parent_dir on the remote server($ip). done"
        sleep 3s
        # copy数据集
        sudo scp -r $dataset_dir root@$ip:$dataset_parent_dir
        sleep 2s
    else
        echo "dataset is ready. skip deploying dataset..."
    fi

    # 在远程服务器上执行FlagPerf目录的备份
    backup_file="${app_name}.bak.${timestamp}.zip"
    echo "prepare to backup. backup_file:  $source_directory/${backup_file}"
    sleep 2s
    ssh $username@$ip "cd $source_directory && zip -r $backup_file ./$app_name/"
    echo "backup $app_name on remote server($ip) succeed."
    sleep 3s

    # 在远程服务器上执行删除目录
    ssh $username@$ip "cd $source_directory && sudo rm -rfv $app_name"
    echo "remote direcotry($app_name) on remote server($ip) succeed."

    # 在远程服务器上删除zip
    ssh $username@$ip "cd $source_directory && rm -fv $zip_filename"

    # 将压缩文件传输到另一台服务器
    scp "$source_directory/$zip_filename" $username@$ip:$source_directory

    # 远程服务器上unzip
    ssh $username@$ip "cd $source_directory && unzip $zip_filename"
done

# 启动训练
cd $source_directory/$app_name/training && sudo python3 run_benchmarks/run.py

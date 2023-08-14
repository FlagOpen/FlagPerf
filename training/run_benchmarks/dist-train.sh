#!/usr/bin/env bash
set +ex 

# scripts for distributed training:
# 0. make sure to correctly config ssh connection for <username>, and <username> has sudo privilege.
# 1. make sure to get same dataset prepared on each rank.
# 2. run dual-server training on the master server.

# 需要填写的信息
ip_list=("xx.xx.xx.xx") # 填写多机训练的ip列表，不包含master server
username="<username>"   # 当前用户

# 压缩文件的目录和文件名
source_directory=$( cd "$(dirname "$0")" && cd ./../../.. ; pwd )

app_name="FlagPerf"
zip_filename="${app_name}.zip"

local_zip=$source_directory/$zip_filename
cd $source_directory && rm -fv $local_zip
cd $source_directory && zip -r $zip_filename $app_name

for ip in "${ip_list[@]}"; do
    # 在远程服务器上执行删除目录
    ssh $username@$ip "cd $source_directory && rm -rfv $app_name"

    # 在远程服务器上删除zip
    ssh $username@$ip "cd $source_directory && rm -fv $zip_filename"

    # 将压缩文件传输到另一台服务器
    scp "$source_directory/$zip_filename" $username@$ip:$source_directory

    # 远程服务器上unzip
    ssh $username@$ip "cd $source_directory && unzip $zip_filename"
done

# 启动测试
cd $source_directory/$app_name/training && sudo python3 run_benchmarks/run.py

#!/bin/bash

SSH="ssh -o StrictHostKeyChecking=no"
SCP="scp -o StrictHostKeyChecking=no"

ssh_pass()
{
    local node="$1"
    local user="$2"
    local pd="$3"
    local port="$4"
    shift 4
    local cmd="$*"

    run_cmd="$node $cmd"
    [ "$user" != "" ] && run_cmd="${user}@$run_cmd"
    [ "$port" != "" ] && run_cmd="-p $port $run_cmd"
    run_cmd="$SSH $run_cmd"
    [ "$pd" != "" ] && run_cmd="sshpass -p ${pd} $run_cmd"
    # echo "run_cmd:$run_cmd"
    $run_cmd || { echo "run sshrun failed node:$node"; return 1; }
}

scp_pass()
{
    local node="$1"
    local user="$2"
    local pd="$3"
    local port="$4"
    local src="$5"
    local target="$6"

    run_cmd="${node}:${target}"
    [ "$user" != "" ] && run_cmd="${user}@$run_cmd"
    run_cmd="-r $src/* ${run_cmd}"
    [ "$port" != "" ] && run_cmd="-P $port $run_cmd"
    run_cmd="${SCP} ${run_cmd}"
    [ "$pd" != "" ] && run_cmd="sshpass -p ${pd} $run_cmd"
    # echo "run_cmd:$run_cmd"
    $run_cmd || { echo "run scp failed node:$node"; return 1; }
}

rscp_pass()
{
    local node="$1"
    local user="$2"
    local pd="$3"
    local port="$4"
    local src="$5"
    local target="$6"

    run_cmd="${node}:${src}/* ${target}"
    [ "$user" != "" ] && run_cmd="${user}@$run_cmd"
    [ "$port" != "" ] && run_cmd="-P $port $run_cmd"
    run_cmd="${SCP} -r ${run_cmd}"
    [ "$pd" != "" ] && run_cmd="sshpass -p ${pd} $run_cmd"
    # echo "run_cmd:$run_cmd"
    $run_cmd || { echo "run rscp failed node:$node"; return 1; }
}

get_cluster_list()
{
    local cluster_config=$1
    cat ${cluster_config} | python3 -c 'import sys,json;[print(node) for node in json.load(sys.stdin)["cluster"].keys()]'
}

get_node_user()
{
    local cluster_config=$1
    local node=$2
    cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster"]['\"${node}\"']["user"])' 2>/dev/null
}

get_node_pd()
{
    local cluster_config=$1
    local node=$2
    cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster"]['\"${node}\"']["pd"])' 2>/dev/null
}

get_node_port()
{
    local cluster_config=$1
    local node=$2
    cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster"]['\"${node}\"']["port"])' 2>/dev/null
}

local_run_cmd()
{
    local cmd="$*"
    (eval $cmd) || { echo "Warn local run '${cmd}'"; return 1; }
}

local_scp_cmd()
{
    local src_path="$2"
    local dst_path="$3"

    [ -d $src_path ] || { echo "Warn src_path:$src_path not exist return";return 1; }

    # rm and mkdir dst path
    # [ -d $dst_path ] && rm -rf $dst_path
    mkdir -p $dst_path

    cp -rf $src_path/* $dst_path 2>/dev/null
    return 0
}

# nodeinfo.json

# {
#  "cluster": {
#    "90.90.66.64": {
#         "user": "root",
#         "pd": "xx",
#          "port": 23,
#    },
#    "90.90.66.66": {
#    }
#  }
# }

# interface function

# 根据参数1串行调用命令 如果失败就返回
# 参数1: 节点信息json文件，包含节点ip和用户名密码信息 如果为空即是本地调用
# 参数其他: 运行的命令
# 样例 cluster_run_cmd_serial "$NODEINFO_FILE" "ifconfig"
cluster_run_cmd_serial()
{
    local node_info_file=$1
    shift 1

    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_run_cmd "$@";return $?; }

    [ -f $node_info_file ] || { echo "$node_info_file not exist ret";return 1; }
    local cmd=$*
    local node_arr=($(get_cluster_list ${node_info_file}))
    local node_count=${#node_arr[@]}

    for ((i=0; i<$node_count; i++)); do {
        local node="${node_arr[$i]}"
        local user=$(get_node_user ${node_info_file} ${node})
        local pd=$(get_node_pd ${node_info_file} ${node})
        local port=$(get_node_port ${node_info_file} ${node})
        local cur_cmd="export SERVER_ID=${i}; ${cmd}"
        ssh_pass "${node}" "${user}" "${pd}" "$port" "${cur_cmd}" || { echo "node:${node} ERROR when executing '${cur_cmd}'"; return 1; }
    }
    done
    return 0
}

# 根据参数1项调用命令，只调用一个节点的命令
# 参数1: 节点信息json文件，包含节点ip和用户名密码信息 如果为空即是本地调用
# 参数其他: 运行命令
# 样例 cluster_run_cmd_single "$NODEINFO_FILE" "ifconfig"
cluster_run_cmd_single()
{
    local node_info_file=$1
    shift 1

    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_run_cmd "$@";return $?; }

    [ -f $node_info_file ] || { echo "$node_info_file not exist ret";return 1; }
    local cmd=$*
    local node_arr=($(get_cluster_list ${node_info_file}))

    local node="${node_arr[0]}"
    local user=$(get_node_user ${node_info_file} ${node})
    local pd=$(get_node_pd ${node_info_file} ${node})
    local port=$(get_node_port ${node_info_file} ${node})
    local cur_cmd="export SERVER_ID=0; ${cmd}"
    ssh_pass "${node}" "${user}" "${pd}" "$port" "${cur_cmd}" || { echo "node:${node} ERROR when executing '${cur_cmd}'"; return 1; }
    return 0
}

# 根据参数1调用命令，并行运行，等待所有命令执行完
# 参数1: 节点信息json文件，包含节点ip和用户名密码信息 如果为空即是本地调用
# 参数其他: 运行命令
# 样例 cluster_run_cmd_parallel "$NODEINFO_FILE" "ifconfig"
cluster_run_cmd_parallel()
{
    local node_info_file=$1
    shift 1
    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_run_cmd "$@";return $?; }

    [ -f $node_info_file ] || { echo "$node_info_file not exist ret";return 1; }
    local cmd=$*
    local node_arr=($(get_cluster_list ${node_info_file}))
    local node_count=${#node_arr[@]}

    local retvalfile=$(mktemp)
    for ((i=0; i<$node_count; i++)); do {
        local node="${node_arr[$i]}"
        local user=$(get_node_user ${node_info_file} ${node})
        local pd=$(get_node_pd ${node_info_file} ${node})
        local port=$(get_node_port ${node_info_file} ${node})
        local cur_cmd="export SERVER_ID=${i}; $cmd"
        ssh_pass "${node}" "${user}" "${pd}" "$port" ${cur_cmd}  || { echo "node:${node} ERROR when executing '${cur_cmd}'"; rm -rf $retvalfile;}
    } &
    done
    wait
    [ -f $retvalfile ] || { echo "run train failed";return 1; }
    rm -rf $retvalfile
}

# 根据参数1拷贝主节点文件夹内容到各运行节点中
# 注意 实际拷贝命令为  cp src_path/* dst_path 且dst_path会执行删除然后重建
# 参数1: 节点信息json文件，包含节点ip和用户名密码信息 如果为空即是本地调用
# 参数2: 主节点的源路径 src_path
# 参数3: 运行节点的源路径 dst_path
# 样例 cluster_scp "${NODEINFO_FILE}" "/home/src" "/home/dst"
cluster_scp()
{
    local node_info_file=$1
    local src_path=$2
    local dst_path=$3

    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_scp_cmd "$@";return $?; }

    local node_arr=($(get_cluster_list ${node_info_file}))
    local node_count=${#node_arr[@]}

    for ((i=0; i<$node_count; i++)); do {
        local node="${node_arr[$i]}"
        local user=$(get_node_user ${node_info_file} ${node})
        local pd=$(get_node_pd ${node_info_file} ${node})
        local port=$(get_node_port ${node_info_file} ${node})
        scp_pass "${node}" "${user}" "${pd}" "$port" "${src_path}" "${dst_path}"  || { echo "scp_pass failed node:$node"; return 1; }
        echo "------------scp  done------${user}@${node}---------------------"
    } done
}

# 根据参数1拷贝各运行节点文件内容到主节点文件夹中
# 注意 实际拷贝命令为  cp src_path/* dst_path 且dst_path会创建
# 参数1: 节点信息json文件，包含节点ip和用户名密码信息 如果为空即是本地调用
# 参数2: 运行节点的源路径 src_path
# 参数3: 主节点的源路径 dst_path
# 样例 cluster_rscp "${NODEINFO_FILE}" "/home/src" "/home/dst"
cluster_rscp()
{
    local node_info_file=$1
    local src_path="$2"
    local dst_path="$3"

    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_scp_cmd "$@";return $?; }

    [ -f $node_info_file ] || { echo "$node_info_file not exist ret";return 1; }

    local node_arr=($(get_cluster_list ${node_info_file}))
    local node_count=${#node_arr[@]}

    for ((i=0; i<$node_count; i++)); do {
        local node="${node_arr[$i]}"
        local user=$(get_node_user ${node_info_file} ${node})
        local pd=$(get_node_pd ${node_info_file} ${node})
        local port=$(get_node_port ${node_info_file} ${node})
        echo "------------------${user}@${node}---------------------"
        rscp_pass "${node}" "${user}" "${pd}" "$port" "${src_path}" "${dst_path}" || { echo "sshpass_rscp failed node:$node"; return 1; }
    } done
}


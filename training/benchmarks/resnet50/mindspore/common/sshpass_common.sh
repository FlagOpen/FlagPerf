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
    echo "run_cmd:$run_cmd"
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
    run_cmd="-r $src ${run_cmd}"
    [ "$port" != "" ] && run_cmd="-P $port $run_cmd"
    run_cmd="${SCP} ${run_cmd}"
    [ "$pd" != "" ] && run_cmd="sshpass -p ${pd} $run_cmd"
    echo "run_cmd:$run_cmd"
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

    run_cmd="${node}:${src} ${target}"
    [ "$user" != "" ] && run_cmd="${user}@$run_cmd"
    [ "$port" != "" ] && run_cmd="-P $port $run_cmd"
    run_cmd="${SCP} -r ${run_cmd}"
    [ "$pd" != "" ] && run_cmd="sshpass -p ${pd} $run_cmd"
    echo "run_cmd:$run_cmd"
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
    [ -d $dst_path ] && rm -rf $dst_path
    mkdir -p $dst_path

    cp -rf "$src_path" $dst_path || { echo "Warn local cp failed";return 1; }
}

# interface function

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
        ssh_pass "${node}" "${user}" "${pd}" "$port" "${cur_cmd}" || { echo "EROOR when executing '${cur_cmd}'"; return 1; }
    }
    done
    return 0
}

cluster_run_cmd_parallel()
{
    node_info_file=$1
    shift 1
    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_run_cmd "$@";return $?; }

    [ -f $node_info_file ] || { echo "$node_info_file not exist ret";return 1; }
    cmd=$*
    node_arr=($(get_cluster_list ${node_info_file}))
    node_count=${#node_arr[@]}

    retvalfile=$(mktemp)
    for ((i=0; i<$node_count; i++)); do {
        node="${node_arr[$i]}"
        user=$(get_node_user ${node_info_file} ${node})
        pd=$(get_node_pd ${node_info_file} ${node})
        port=$(get_node_port ${node_info_file} ${node})
        cur_cmd="export SERVER_ID=${i}; $cmd"
        echo "正在登陆${user}@${node}，SERVER_ID为${i}, 进入后执行的命令为${cur_cmd}"
        ssh_pass "${node}" "${user}" "${pd}" "$port" ${cur_cmd}  || { echo "run scp failed node:$node"; rm -rf $retvalfile; }
    }&
    done
    logger_Info "now wait run cmd done"
    wait
    [ -f $retvalfile ] || { echo "run train failed";return 1; }
    rm -rf $retvalfile
    logger_Info "now run cmd done finish"
}

sshpass_scp_cmd()
{
    node_info_file=$1

    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_scp_cmd "$@";return $?; }

    src_path=$2
    dst_path=$3
    node_arr=($(get_cluster_list ${node_info_file}))
    node_count=${#node_arr[@]}

    for ((i=0; i<$node_count; i++)); do {
        node="${node_arr[$i]}"
        user=$(get_node_user ${node_info_file} ${node})
        pd=$(get_node_pd ${node_info_file} ${node})
        port=$(get_node_port ${node_info_file} ${node})
        echo "------------scp ------${user}@${node}---------------------"
        ssh_pass "${node}" "${user}" "${pd}" "$port" "rm -rf ${dst_path};mkdir -p ${dst_path}" || { echo "sshpass_scp failed node:$node"; return 1; }
        scp_pass "${node}" "${user}" "${pd}" "$port" "${src_path}" "${dst_path}"  || { echo "sshpass_scp failed node:$node"; return 1; }
        echo "------------scp  done------${user}@${node}---------------------"
    } done
}

sshpass_rscp_cmd()
{
    node_info_file=$1

    # clusterconfig file not set as local mode
    [ "$node_info_file" == "" ] && { local_scp_cmd "$@";return $?; }

    [ -f $node_info_file ] || { echo "$node_info_file not exist ret";return 1; }

    src_path="$2"
    dst_path="$3"
    node_arr=($(get_cluster_list ${node_info_file}))
    node_count=${#node_arr[@]}

    for ((i=0; i<$node_count; i++)); do {
        node="${node_arr[$i]}"
        user=$(get_node_user ${node_info_file} ${node})
        pd=$(get_node_pd ${node_info_file} ${node})
        port=$(get_node_port ${node_info_file} ${node})
        echo "------------------${user}@${node}---------------------"
        # ssh_pass ${node} ${user} ${pd} "rm -rf ${dst_path}"
        rscp_pass "${node}" "${user}" "${pd}" "$port" "${src_path}" "${dst_path}" || { echo "sshpass_rscp failed node:$node"; return 1; }
    } done
}
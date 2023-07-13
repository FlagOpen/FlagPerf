
get_node_podname()
{
    local rank_table_file=$1
    local server_id=$2
    cat ${rank_table_file} | python3 -c 'import sys,json;print(json.load(sys.stdin)["group_list"][0]["instance_list"]['${server_id}']["pod_name"])' #2>/dev/null
}

# 通用检测 在主节点上检测环境是否正常
check_env_common()
{
    : "${RANK_SIZE?RANK_SIZE not set}"
    : "${DEVICE_NUM?DEVICE_NUM not set}"

    # check ranktable set
    [[ $RANK_SIZE -eq 1 ]] || : "${RANK_TABLE_FILE?RANK_TABLE_FILE not set}"
    [[ $RANK_SIZE -eq 1 ]] && [[ -n "$RANK_TABLE_FILE" ]] && { echo "ranksize=1 should not set RANK_TABLE_FILE";return 1; }

    : "${PYTHON_COMMAND?PYTHON_COMMAND not set}"

    # check nodeinfofile exist
    [[ $RANK_SIZE -le 8 ]] || check_file_valid "${NODEINFO_FILE}" || { echo "nodeinfofile:${NODEINFO_FILE} not valid" ; return 1; }

    # check basic command of the main node
    if [ -f "$NODEINFO_FILE" ];then
        check_command_exist ssh || { echo "ssh running failed" ; return 1; }
        echo "ssh running successfully"
        check_command_exist sshpass || { echo "sshpass running failed" ; return 1; }
        echo "sshpass running successfully"
    fi
    return 0
}

# 通用检测 主要检测 PYTHON_COMMAND RANK_SIZE和RANK_TABLE
node_common_check()
{
    local pythoncmd="$1"
    local ranksize="$2"
    local ranktable="$3"
    check_command_exist ${pythoncmd} || { logger_Warn "python:$pythoncmd running failed" ; return 1; }
    echo "${pythoncmd} running successfully"

    if [ ${ranksize} != 1 ]; then
        check_file_valid "$ranktable" || { logger_Warn "RANK_TABLE_FILE:${ranktable} not valid path" ; return 1; }
        echo "RANK_TABLE_FILE path valid"
    fi
    return 0
}

# 通用训练函数调用
# 必须依赖变量包括 WORK_PATH get_train_cmd 会做检查
# 参数1 是否绑核 如果需要 传入 "true"
# 参数2 是否老的ranktable 如果需要 传入 "true"

function node_common_train()
{
    [ -d "$WORK_PATH" ] || { echo "not exit WORK_PATH return";return 1; }
    [ "$(type -t get_train_cmd)" == 'function' ] || { echo "not exist get_train_cmd func return";return 1; }

    bindcore=$([ "$1" == "true" ] && echo "true" || echo "false")
    oldranktable=$([ "$2" == "true" ] && echo "true" || echo "false")

    # get server node id default is 0
    : "${SERVER_ID:=0}"
    # get rank start index
    if [[ $DEVICE_NUM == 1  && $RANK_SIZE == 1 ]];then
        : "${SINGLE_CARD_INDEX:=0}"
        RANK_START=$SINGLE_CARD_INDEX
    else
        # get rank start index
        RANK_START=`expr ${SERVER_ID} \* $DEVICE_NUM`
    fi
    # set bind core
    [ $bindcore == "true" ] && { cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`; avg=`expr $cpus \/ $DEVICE_NUM`; gap=`expr $avg \- 1`; }
    if [ $oldranktable == "true" ];then
        podname=$(get_node_podname ${RANK_TABLE_FILE} ${SERVER_ID})
    fi

    retvalfile=$(mktemp)
    for((i=0;i<${DEVICE_NUM};i++));do
    {
        index=$[i+RANK_START]
        export DEVICE_ID=${i}
        # old ranktable should set DEVICE_INDEX and new ranktable should set RANK_ID
        if [ $oldranktable == "true" ];then
            export DEVICE_INDEX=$[i+RANK_START]
            export RANK_ID=$podname
        else
            export RANK_ID=$[i+RANK_START]
            # 应该是device_id吧
            export ASCEND_DEVICE_ID=$DEVICE_ID
            export DEVICE_INDEX=$DEVICE_ID
            export RANK_INDEX=$SERVER_ID
        fi
        # clear and create path.
        RUN_PATH="$WORK_PATH/train_parallel$index"
        mkdir -p $RUN_PATH; cd $RUN_PATH;
        # if bindcore should get cmdopt for cores
        [ $bindcore == "true" ] && { start=`expr $i \* $avg`; end=`expr $start \+ $gap`; cmdopt=$start"-"$end; }
        # call out func get run cmd
        get_train_cmd
        logger_Info "start training for SERVER_ID:$SERVER_ID rank $index, device $DEVICE_ID begin cmd:$train_run_cmd"
        # if bindcore add taskset
        [ $bindcore == "true" ] && train_run_cmd="taskset -c $cmdopt $train_run_cmd"
        # call cmd
        eval $train_run_cmd | tee -a $RUN_PATH/train.log 2>&1 || { logger_Warn "train failed rank $index, device $DEVICE_ID failed:$?"; rm -rf $retvalfile; }
    } &
    done
    logger_Info "Waiting for the training process of SERVER_ID:${SERVER_ID} to finish"
    wait
    [ -f $retvalfile ] || { logger_Warn "run train failed";return 1; }
    logger_Info "SERVER_ID:${SERVER_ID} training finished"
}

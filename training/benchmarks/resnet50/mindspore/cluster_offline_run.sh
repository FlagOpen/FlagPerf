#!/bin/bash
CUR_PATH=$(dirname $(readlink -f "$0"))
export CODE_PATH=$CUR_PATH
export BASE_PATH=$(cd "$CUR_PATH/../";pwd)

. $CODE_PATH/common/common.sh
. $CODE_PATH/common/log_util.sh
. $CODE_PATH/common/cluster_common.sh
. $CODE_PATH/common/node_common.sh

# env check
check_env()
{
    # base check
    check_env_common || { logger_Warn "check env common failed:$?";return 1; }

    # model info check
    : "${TRAIN_DATA_PATH?TRAIN_DATA_PATH not set}"
    : "${EVAL_DATA_PATH?EVAL_DATA_PATH not set}"

    # check env of each node
    cmd="export WORK_PATH=$WORK_PATH;
        bash $WORK_PATH/run_node.sh check ${WORK_PATH}/config/$CONFIG_FILE"
    cluster_run_cmd_serial "$NODEINFO_FILE" ${cmd} || { return 1; }
}

init()
{
    logger_Info "-------------------------------- init start --------------------------------"
    # set nodes result path
    export WORK_PATH=$CUR_PATH
    export RESULT_PATH=${LOG_DIR}
    export TRAIN_DATA_PATH=${DATA_DIR}/train/
    export EVAL_DATA_PATH=${DATA_DIR}/val/
    export RANK_SIZE=$NPROC
    export DEVICE_NUM=$NPROC


    cd $WORK_PATH
    CONFIG_FILE="config.sh"
    source ./config/$CONFIG_FILE || { logger_Warn "source file failed:$?";return 1; }

    check_env || { logger_Warn "env check failed'" ; return 1; }
    logger_Info "-------------------------------- init end --------------------------------"
}

run_train()
{
    logger_Info "-------------------------------- train start --------------------------------"
    cmd="export WORK_PATH=$WORK_PATH;
        export RESULT_PATH=$RESULT_PATH;
        source $WORK_PATH/config/$CONFIG_FILE;
        bash $WORK_PATH/run_node.sh train"

    cluster_run_cmd_parallel "${NODEINFO_FILE}" ${cmd} || { logger_Warn "run train failed"; return 1; }
    logger_Info "-------------------------------- train end --------------------------------"
}

run_eval()
{
    logger_Info "-------------------------------- eval start --------------------------------"
    cmd="source $WORK_PATH/config/$CONFIG_FILE;
        export WORK_PATH=$WORK_PATH;
        export RESULT_PATH=$RESULT_PATH;
        bash $WORK_PATH/run_node.sh eval"
    cluster_run_cmd_single "${NODEINFO_FILE}" ${cmd} || { logger_Warn "run eval failed"; return 1; }
    logger_Info "-------------------------------- eval end --------------------------------"
}

get_result()
{
    logger_Info "-------------------------------- get_result start --------------------------------"

    cmd="mkdir -p ${RESULT_PATH}"
    cluster_run_cmd_serial "$NODEINFO_FILE" ${cmd} || { logger_Warn "mkdir resultpath failed"; return 1; }

    cluster_rscp "${NODEINFO_FILE}" ${RESULT_PATH} ${RESULT_PATH}
    ${PYTHON_COMMAND} ${CODE_PATH}/common/calc_result.py ${RESULT_PATH} ${RANK_SIZE}
    
    [ -d $BASE_PATH/result ] && cp ${RESULT_PATH}/* -rf  $BASE_PATH/result/
    logger_Info "-------------------------------- get_result end --------------------------------"
}

LONGOPS="log_dir:,data_dir:,nproc:"
PARSED=$(getopt --options=hi:o --longoptions="$LONGOPS" --name "$0" -- "$@")
eval set -- "$PARSED"
while true; do
    case "$1" in
        -l|--log_dir)
            LOG_DIR="$2"
            shift=2
            ;;
        -d|--data_dir)
            DATA_DIR="$2"
            shift=2
            ;;
        -n|--nproc)
            NPROC="$2"
            shift=2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknow option: $1"
            exit 1
            ;;
    esac
done

train()
{
    export RANK_SIZE=$NPROC
    export DEVICE_NUM=$NPROC
    export WORK_PATH=$CUR_PATH
    export TRAIN_DATA_PATH=${DATA_DIR}/train/
    export EVL_DATA_PATH=${DATA_DIR}/val/
    cd $WORK_PATH
    export RESULT_PATH=${LOG_DIR}/
    source ./config/config.sh
    bash -x ./run_node.sh train
}

train

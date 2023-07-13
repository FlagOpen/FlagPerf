#!/bin/bash
CUR_PATH=$(dirname $(readlink -f "$0"))
export CODE_PATH=$CUR_PATH
export BASE_PATH=$(cd "$CUR_PATH/../";pwd)

. $CODE_PATH/common/common.sh
. $CODE_PATH/common/log_util.sh
. $CODE_PATH/common/cluster_common.sh
. $CODE_PATH/common/node_common.sh

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

function get_train_cmd()
{
    [[ $RANK_SIZE -gt 1 ]] && DISTRUTE_ENABLE="True" || DISTRUTE_ENABLE="False"
    # 基准代码r2.0.0版本中训练配置文件resnet50_imagenet2012_Boost_config.yaml中，将训练参数output_path改为output_dir
    CONFIG_PATH=$WORK_PATH/config/${CONFIG_FILE}
    isexisted=`cat $CONFIG_PATH |grep "output_dir" |grep -v grep |awk -F= 'NR==1{print $NF}'`
    if [ ! -n "$isexisted" ]; then
        OUTPUT_PARA_NAME="output_path"
    else
        OUTPUT_PARA_NAME="output_dir"
    fi

    train_run_cmd="${PYTHON_COMMAND} -u $WORK_PATH/code/train.py \
        --run_distribute=$DISTRUTE_ENABLE \
        --data_path=${TRAIN_DATA_PATH} \
        --device_num=${DEVICE_NUM}  \
        --epoch_size=${EPOCH_SIZE}  \
        --$OUTPUT_PARA_NAME="$RUN_PATH"  \
        --save_checkpoint=True  \
        --save_checkpoint_epochs=${EPOCH_SIZE} \
        --config_path=$CONFIG_PATH
        "
        # for mindspore1.5
        export ENV_FUSION_CLEAR=1
        export ENV_SINGLE_EVAL=1
        export SKT_ENABLE=1
}

function get_eval_cmd()
{
    eval_run_cmd="${PYTHON_COMMAND} -u $WORK_PATH/code/eval.py \
         --data_path=${EVAL_DATA_PATH} \
         --config_path=$WORK_PATH/config/${CONFIG_FILE} \
         --checkpoint_file_path=${CHECKPOINT_PATH}"
    return 0
}

function node_init()
{
    export PYTHONPATH=$PYTHONPATH:$WORK_PATH
    # for eval env set
    [ $1 == "eval" ] && { export RANK_SIZE=1; export DEVICE_ID=0; : "${SINGLE_CARD_INDEX:=0}";export RANK_ID=$SINGLE_CARD_INDEX; unset RANK_TABLE_FILE; }
    [[ -z "$RESULT_PATH" ]] || { mkdir -p $RESULT_PATH; }
}

function node_check()
{
    CONFIG_FILE_PATH=$1
    source $CONFIG_FILE_PATH

    node_common_check "${PYTHON_COMMAND}" "${RANK_SIZE}" "$RANK_TABLE_FILE" || { logger_Warn "node common check failed" ; return 1; }

    check_mindspore_run_ok ${PYTHON_COMMAND} || { logger_Warn "mindspore running failed" ; return 1; }
    logger_Debug "mindspore running successfully"

    check_path_valid "${TRAIN_DATA_PATH}" || { logger_Warn "TRAIN_DATA_PATH:${TRAIN_DATA_PATH} not valid path" ; return 1; }
    logger_Debug "TRAIN_DATA_PATH is valid"

    check_path_valid "${EVAL_DATA_PATH}" || { logger_Warn "EVAL_DATA_PATH:${EVAL_DATA_PATH} not valid path" ; return 1; }
    logger_Debug "EVAL_DATA_PATH is valid"
}

function node_train()
{
    node_common_train "true" "false" || { logger_Warn "run train failed" ; return 1; }
}

function node_eval()
{
    CHECKPOINT_PATH=`find ${WORK_PATH}/train_parallel$RANK_ID/ -name "*.ckpt" | xargs ls -t | awk 'NR==1{print}'`
    [ -f $CHECKPOINT_PATH ] || { logger_Warn "CHECKPOINT_PATH:${CHECKPOINT_PATH} not valid path" ; return 1; }
    cp $CHECKPOINT_PATH  $RESULT_PATH/
    RUN_PATH=$WORK_PATH/train_parallel$RANK_ID
    cd $RUN_PATH
    get_eval_cmd
    echo "start eval RUN_PATH:${RUN_PATH} SERVER_ID:$SERVER_ID rank $RANK_ID device $DEVICE_ID begin cmd:${eval_run_cmd}"
    $eval_run_cmd || { echo "run eval node error ret:$?"; return 1; }
    return 0
}

main()
{
    export RANK_SIZE=$NPROC
    export DEVICE_NUM=$NPROC
    export WORK_PATH=$CUR_PATH
    export TRAIN_DATA_PATH=${DATA_DIR}/train/
    export EVL_DATA_PATH=${DATA_DIR}/val/
    cd $WORK_PATH
    export RESULT_PATH=${LOG_DIR}/
    source ./config/config.sh

    node_train "$@" || { logger_Warn "run_node_train failed"; return 1; }
}

main "$@"
exit $?


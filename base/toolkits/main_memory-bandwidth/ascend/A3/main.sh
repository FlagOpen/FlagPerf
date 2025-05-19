source /usr/local/Ascend/toolbox/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
LOG_PATH=`pwd`
ascend-dmi --bw -t d2d -q > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(awk 'NR>29 && NR<31 {print $3}' ${LOG_PATH}/test_result.log)
RESULT_INT=$(printf "%.0f" "$RESULT")
RESULT_A3=$(expr $RESULT_INT \* 2)
echo "[FlagPerf Result] main_memory-bandwidth=${RESULT_A3} GB/s"
rm -rf ${LOG_PATH}/test_result.log

source /usr/local/Ascend/toolbox/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
LOG_PATH=`pwd`
npu-smi info -t memory -i 0 -c 0 > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(grep "HBM Capacity(MB)" ${LOG_PATH}/test_result.log | awk '{print $NF}')
RESULT_A3=$(expr $RESULT \* 2)
echo "[FlagPerf Result] main_memory-capacity=${RESULT_A3} MiB"
rm -rf ${LOG_PATH}/test_result.log

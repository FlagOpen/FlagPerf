source /usr/local/Ascend/toolbox/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
LOG_PATH=`pwd`
ascend-dmi -f -t int8 -q > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(awk 'NR>3 && NR<5 {print $4}' ${LOG_PATH}/test_result.log)
echo "[FlagPref Result] computation-INT8=$RESULT TFLOPS"
rm -rf ${LOG_PATH}/test_result.log
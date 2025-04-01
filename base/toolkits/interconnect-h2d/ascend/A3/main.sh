source /usr/local/Ascend/toolbox/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
LOG_PATH=`pwd`
ascend-dmi --bw -t h2d -s 4294967296 --et 100 -q > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(awk 'NR>5 && NR<7 {print $4}' ${LOG_PATH}/test_result.log)
echo "[FlagPref Result] interconnect-h2d bandwidth=$RESULT GB/s"
rm -rf ${LOG_PATH}/test_result.log
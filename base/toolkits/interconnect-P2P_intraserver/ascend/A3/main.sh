source /usr/local/Ascend/toolbox/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
LOG_PATH=`pwd`
ascend-dmi --bw -t p2p -m card -q > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(awk 'NR>21 && NR<23 {print $2}' ${LOG_PATH}/test_result.log)
echo "[FlagPref Result] interconnect-P2P_intraserver-bandwidth=${RESULT} GB/s"
rm -rf ${LOG_PATH}/test_result.log
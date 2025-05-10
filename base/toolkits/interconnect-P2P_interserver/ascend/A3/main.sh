source /usr/local/Ascend/toolbox/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
LOG_PATH=`pwd`
ascend-dmi -bw -t p2p --sp 0 -q --ip 127.0.0.1 --spp /home/zhiyuan/share_path --hip 127.0.0.2 -m card > ${LOG_PATH}/test_result.log 2>&1
RESULT=$(awk 'NR>13 && NR<15 {print $3}' ${LOG_PATH}/test_result.log)
echo "[FlagPerf Result] interconnect-P2P_interserver-bandwidth=${RESULT} GB/s"
rm -rf ${LOG_PATH}/test_result.log

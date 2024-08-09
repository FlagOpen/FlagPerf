export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
cnvs -r mlulink -c `pwd`/cnvs.example.yml 2>&1 | tee ${LOG_PATH}
device0_1=$(sed -n '58p' "$LOG_PATH" | awk '{print $7}')
device1_0=$(sed -n '62p' "$LOG_PATH" | awk '{print $5}')
result=$(python3 -c "print(float($device0_1)*0.5 + float($device1_0)*0.5)")
echo "[FlagPerf Result]interconnect-P2P_intraserver-bandwidth=${result} GB/s"
rm -rf cnvs_stats ${LOG_PATH} #删除缓存文件
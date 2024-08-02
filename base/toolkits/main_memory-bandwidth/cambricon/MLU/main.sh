export MLU_VISIBLE_DEVICES=0
LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
cnvs -r memory_bandwidth -c `pwd`/cnvs.example.yml 2>&1 | tee ${LOG_PATH}
value=$(grep "read" "$LOG_PATH" | awk '{print $2}')
echo "[FlagPerf Result]main_memory-bandwidth=${value} GB/s"
rm -rf cnvs_stats ${LOG_PATH} #删除缓存文件
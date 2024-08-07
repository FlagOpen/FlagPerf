LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
/usr/local/neuware/bin/allreduce \
    --warmup_loop 20  \
    --thread 8 \
    --loop 2000 \
    --mincount 1 \
    --maxcount 512M \
    --multifactor 2 \
    --async 1 \
    --block 0 2>&1 | tee ${LOG_PATH}
data=$(tail -n 2 ${LOG_PATH} | awk '{print $11 }')
result=$(python3 -c "print(float($data) * 2)")
echo "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=$result GB/s"
rm -rf ${LOG_PATH} #删除缓存文件
#!/bin/bash
export MLU_VISIBLE_DEVICES=0
LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
cnvs -r pcie -c `pwd`/cnvs.example.yml 2>&1 | tee ${LOG_PATH}
bandwidth=$(sed -n '17p' "$LOG_PATH" | awk '{print $5}')
echo "[FlagPerf Result] interconnect-h2d bandwidth=$bandwidth GB/s"
rm -rf cnvs_stats ${LOG_PATH} #删除缓存文件
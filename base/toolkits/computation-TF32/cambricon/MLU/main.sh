#!/bin/bash
export MLU_VISIBLE_DEVICES=0
LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
cnvs -r matmul_performance -c `pwd`/cnvs.example.yml 2>&1 | tee ${LOG_PATH}
value=$(grep -o 'matmul performance(GOPS): [0-9.]\+' ${LOG_PATH} )
number=$(echo $value | grep -o '[0-9.]\+')
result=$(python3 -c "print(float($number) / 1000)")
echo "[FlagPerf Result] computation-TF32=$result TFLOPS"
rm -rf cnvs_stats ${LOG_PATH} #删除缓存文件

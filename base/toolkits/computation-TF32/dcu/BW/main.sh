LOG_PATH=$(pwd)/$(ip a | grep -w 'inet' | grep 'global' | sed 's/.*inet //;s/\/.*//' | awk 'NR==1{print $1}')_run_log

export PATH=/opt/hyqual_v3.0.3:${PATH}
run 7 2>&1 | tee ${LOG_PATH}

data=$(grep 'peak tf32gemm :'  ${LOG_PATH} | awk '{print $4}' | sort -nr | head -n1)
echo "[FlagPerf Result]computation-TF32=$data TFLOPS"
rm -rf ${LOG_PATH}

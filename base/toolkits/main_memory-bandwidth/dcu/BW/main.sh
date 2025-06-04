LOG_PATH=$(pwd)/$(ip a | grep -w 'inet' | grep 'global' | sed 's/.*inet //;s/\/.*//' | awk 'NR==1{print $1}')_run_log

export PATH=/opt/hyqual_v3.0.3:${PATH}
run 6 2>&1 | tee ${LOG_PATH}

data=$(grep 'HCU.*mem bandwidth' ${LOG_PATH} | \
  sed -E 's/.* ([0-9]+\.[0-9]+)GB\/s.*/\1/' | \
  sort -gr | \
  head -n1 )

echo "[FlagPerf Result]main_memory-bandwidth=$data GB/s"
rm -rf ${LOG_PATH}
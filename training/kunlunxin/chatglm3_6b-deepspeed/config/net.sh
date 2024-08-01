export CUDA_DEVICE_MAX_CONNECTIONS=1
ipcs -m | awk '/0x/ {print $2}' | xargs -n 1 ipcrm shm
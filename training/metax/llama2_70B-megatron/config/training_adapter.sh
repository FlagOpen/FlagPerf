export MALLOC_THRESHOLD=99
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export MACA_SMALL_PAGESIZE_ENABLE=1
export MALLOC_THRESHOLD=95
export MCPYTORCH_DISABLE_PRINT=1

export MCCL_NET_GDR_LEVEL=7
#ENV MCCL_MIN_NCHANNELS=16
export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1

export MHA_USE_BLAS=ON
export LD_LIBRARY_PATH=/root/FWD_76_BWD_79:${LD_LIBRARY_PATH}
export SET_DEVICE_NUMA_PREFERRED=1

export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 6
"
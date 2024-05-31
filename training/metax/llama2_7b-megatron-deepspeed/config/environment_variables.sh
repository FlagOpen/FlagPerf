export MACA_PATH=/opt/maca-2.18.0.8

export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}

unset CUDA_HOME
export PATH=${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}

#MACA-PyTorch envs
export ISU_FASTMODEL=1 # must be set, otherwise may induce precision error
export USE_TDUMP=OFF # optional, use to control whether generating debug file
export TMEM_LOG=OFF # optional, use to control whether generating debug file
export DEBUG_ITRACE=0 # optional, use to control whether generating debug file

export MACA_SMALL_PAGESIZE_ENABLE=1
export MALLOC_THRESHOLD=99
export MCPYTORCH_DISABLE_PRINT=1
export SET_DEVICE_NUMA_PREFERRED=1

export MCCL_NET_GDR_LEVEL=7
export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1

export MHA_USE_BLAS=ON
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCCL_IB_GID_INDEX=3
export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export LD_LIBRARY_PATH='/root/FWD_76_BWD_79':$LD_LIBRARY_PATH

# =================================================
# Export variables
# =================================================

export CONTAINER_MOUNTS="--gpus all"
NVCC_ARGUMENTS="-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr -ftemplate-depth=1024 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80"
if [[ "$PYTORCH_BUILD_VERSION" == 1.8* ]]; then
    NVCC_ARGUMENTS="${NVCC_ARGUMENTS} -D_PYTORCH18"
fi
export NVCC_ARGUMENTS
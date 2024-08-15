echo "KUNLUNXIN ENV.SH start"

source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda
export TRITON_XPU_ARCH=3
export CUDART_DUMMY_REGISTER=1

echo "KUNLUNXIN ENV.SH end"

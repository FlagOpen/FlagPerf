echo "KUNLUNXIN ENV.SH start"

source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda
export TRITON_LOCAL_VALUE_MAX=2048

echo "KUNLUNXIN ENV.SH end"

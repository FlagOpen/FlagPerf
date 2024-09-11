echo "KUNLUNXIN ENV.SH start"

source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda
export TRITONXPU_BUFFER_SIZE=128

echo "KUNLUNXIN ENV.SH end"

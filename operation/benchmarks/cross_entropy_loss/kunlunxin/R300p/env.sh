echo "KUNLUNXIN ENV.SH start"

source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda
export XPU_enable_reorder=1

echo "KUNLUNXIN ENV.SH end"

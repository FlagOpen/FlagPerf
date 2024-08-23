echo "KUNLUNXIN ENV.SH start"

source /root/miniconda/etc/profile.d/conda.sh && conda activate python38_torch201_cuda
export Triton_big_instcombine=1000

echo "KUNLUNXIN ENV.SH end"

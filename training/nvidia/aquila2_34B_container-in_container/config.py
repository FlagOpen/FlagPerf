# =========================================================
# network
# =========================================================
SSH_PORT = "60022"

net_cmd = "export CUDA_DEVICE_MAX_CONNECTIONS=1;export NCCL_SOCKET_IFNAME=enp;export NCCL_IB_DISABLE=0;export NCCL_IB_CUDA_SUPPORT=1;export NCCL_IB_GID_INDEX=0;export NCCL_IB_HCA=mlx5_2,mlx5_5;export NCCL_DEBUG=debug"

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "312000000000000"

# =========================================================
# env attribute
# =========================================================
env_cmd = "export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

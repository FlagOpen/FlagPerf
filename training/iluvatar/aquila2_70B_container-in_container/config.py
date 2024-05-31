# =========================================================
# network
# =========================================================
SSH_PORT = "10020"

net_cmd = "export CUDA_DEVICE_MAX_CONNECTIONS=1; \
        export NCCL_SOCKET_IFNAME=`ip -4 addr show | grep inet | grep 10.31.12. | sed -e 's/^.*global *//g'`; \
        export GLOO_SOCKET_IFNAME=`ip -4 addr show | grep inet | grep 10.31.12. | sed -e 's/^.*global *//g'`; \
        export NCCL_NET_SHARED_BUFFERS=0; \
        export NCCL_FORCESYNC_DISABLE=1; \
        export UMD_CCLINLASTCE=1; \
        export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1; \
        export NCCL_DEBUG=TRACE; \
        export NCCL_ALGO=Ring; \
        export OMP_NUM_THREADS=4; \
        export NCCL_USE_DIRECT=1"
# =========================================================
# chip attribute
# =========================================================
flops_16bit = "192000000000000"

# =========================================================
# env attribute
# =========================================================
env_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ===========!!!!!!FOR 70B 4GPU TESTING!!!!!!==============
# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = 2
PIPELINE_PARALLEL = 32

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 1
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# gradient_accu_steps is the same as flagscale aquila-70B(44)
GLOBAL_BATCHSIZE = 2048
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

# ===========!!!!!!FOR 70B 7GPU TESTING!!!!!!==============
# # =========================================================
# # parallel
# # =========================================================
# TENSOR_PARALLEL = 2
# PIPELINE_PARALLEL = 28

# # =========================================================
# # batch
# # =========================================================
# MICRO_BATCHSIZE = 2
# # globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# # gradient_accu_steps is the same as flagscale aquila-70B(44)
# GLOBAL_BATCHSIZE = 2048
# # 2k for aquila2-7B, 4k for aquila2-34B and 70B
# SEQLENGTH = 4096

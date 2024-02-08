# =========================================================
# network
# =========================================================
SSH_PORT = "8888"

# =========================================================
# env attribute
# =========================================================
env_cmd = "source /root/miniconda3/bin/activate Aquila; source /usr/local/Ascend/driver/bin/setenv.bash; source /usr/local/Ascend/ascend-toolkit/set_env.sh"

DATA_DIR = "/data/wudao_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/mnt/baai_test/wspace/FlagScale"
# 1B tokens for nnodes=1, model=7B
# 1000 steps * SEQLENGTH (4096) * GLOBAL_BATCHSIZE (64) = 262144000
TRAINING_TOKENS = 262144000

# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = 8
PIPELINE_PARALLEL = 2

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 2
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# gradient_accu_steps is the same as flagscale aquila-7B(9)
GLOBAL_BATCHSIZE = 64
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "313000000000000"

# =========================================================
# network
# =========================================================
net_cmd ="export HCCL_CONNECT_TIMEOUT=3600;export HCCL_EXEC_TIMEOUT=0;export GLOO_SOCKET_IFNAME=bond0;export HCCL_SOCKET_IFNAME=bond0;export CUDA_DEVICE_MAX_CONNECTIONS=1"
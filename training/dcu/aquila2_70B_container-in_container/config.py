# =========================================================
# network
# =========================================================
SSH_PORT = "60022"

net_cmd = "export CUDA_DEVICE_MAX_CONNECTIONS=1;export NCCL_SOCKET_IFNAME=ib0;export NCCL_IB_DISABLE=0;export NCCL_DEBUG=debug;export HSA_FORCE_FINE_GRAIN_PCIE=1;export NCCL_IB_TIMEOUT=22;export OMP_NUM_THREADS=1"

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "20480000000000"

# =========================================================
# env attribute
# =========================================================

env_cmd = ""

# =========================================================
# data
# =========================================================
DATA_DIR = "/data/aquila2_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/data/aquila2_pretrain/FlagScale"
# 1B tokens for nnodes=1, model=7B
TRAINING_TOKENS = 30000000

# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = 8
PIPELINE_PARALLEL = 4

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 1
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# gradient_accu_steps is the same as flagscale aquila-70B(44)
GLOBAL_BATCHSIZE = 44
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096
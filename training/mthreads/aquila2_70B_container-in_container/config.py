# =========================================================
# network
# =========================================================
SSH_PORT = "62216"
net_cmd ="export CUDA_DEVICE_MAX_CONNECTIONS=1;export NCCL_PROTOS=2;export MUSA_KERNEL_TIMEOUT=3600000;export OMP_NUM_THREADS=4"

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "98000000000000"

# =========================================================
# env attribute
# =========================================================
env_cmd = "export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH"

# =========================================================
# data
# =========================================================
DATA_DIR = "/home/dist/zhiyuan/wudao_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/home/dist/xiaokang/FlagScale"
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
# gradient_accu_steps is the same as flagscale aquila-7B(9)
GLOBAL_BATCHSIZE = 72
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

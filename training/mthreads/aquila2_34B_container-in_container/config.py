# =========================================================
# network
# =========================================================
SSH_PORT = "62216"
net_cmd ="export CUDA_DEVICE_MAX_CONNECTIONS=1;export NCCL_PROTOS=2;export MUSA_KERNEL_TIMEOUT=1200000;export OMP_NUM_THREADS=4"

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "98000000000000"

# =========================================================
# env attribute
# =========================================================
env_cmd = "export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH"

DATA_DIR = "/home/dist/dataset/wudao_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/home/dist/zhiyuan-test/FlagScale"
# 1B tokens for nnodes=1, model=7B
# TRAINING_TOKENS = 385875968 # for 256 gpu 23 steps, 23 * 4096(gbs) * 4096(seq_len) = 385875968
# TRAINING_TOKENS = 69206016 # for 48 gpu
TRAINING_TOKENS = int(os.environ.get('TRAINING_TOKENS', default=69206016))
# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = 4
PIPELINE_PARALLEL = 4

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 1
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# gradient_accu_steps is the same as flagscale aquila-7B(9)
# NUM_MICROBATCHES=256
import os
WORLD_SIZE=int(os.environ.get('MLFLOW_WORKER_TOTAL_GPUNUM'))
print("WORLD_SIZE:{}".format(WORLD_SIZE))

DP_SIZE = int(WORLD_SIZE / (TENSOR_PARALLEL * PIPELINE_PARALLEL))

NUM_MICROBATCHES = 256

GLOBAL_BATCHSIZE = MICRO_BATCHSIZE * NUM_MICROBATCHES * DP_SIZE
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

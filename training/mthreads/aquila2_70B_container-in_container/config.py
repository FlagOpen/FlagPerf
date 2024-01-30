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
env_cmd = "export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH;export CUDA_DEVICE_MAX_CONNECTIONS=1"
import os
# env_cmd = "export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH;export CUDA_DEVICE_MAX_CONNECTIONS=1;" + "export MLFLOW_TRACKING_URI=" + os.getenv("MLFLOW_TRACKING_URI") + "; export MLFLOW_RUN_ID=" + os.getenv("MLFLOW_RUN_ID") + "; export MCCFLOW_TRAINING_PROGRESS_URI=" + os.getenv("MCCFLOW_TRAINING_PROGRESS_URI")


DATA_DIR = "/home/dist/dataset/wudao_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/home/dist/zhiyuan-test/FlagScale"
# 1B tokens for nnodes=1, model=7B
# TRAINING_TOKENS = 10000000000
# TRAINING_TOKENS = 176160768 # for 256 gpus, 14 steps 3072 * 14 * 4096 = 176160768
# TRAINING_TOKENS = 88080384 # for 128 gpus
TRAINING_TOKENS = 20000000000 # for 1024 gpus
# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = int(os.environ.get('TP_SIZE', default=4))
PIPELINE_PARALLEL = int(os.environ.get('PP_SIZE', default=8))

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 1
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# gradient_accu_steps is the same as flagscale aquila-7B(9)
# WORLD_SIZE=96
import os
WORLD_SIZE = int(os.environ.get('MLFLOW_WORKER_TOTAL_GPUNUM', default=288))
print("WORLD_SIZE:{}".format(WORLD_SIZE))

DP_SIZE = int(WORLD_SIZE / (TENSOR_PARALLEL * PIPELINE_PARALLEL))

NUM_MICROBATCHES = 384

GLOBAL_BATCHSIZE = MICRO_BATCHSIZE * NUM_MICROBATCHES * DP_SIZE
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

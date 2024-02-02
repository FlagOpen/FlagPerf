# =========================================================
# env attribute
# =========================================================
env_cmd = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"

DATA_DIR = "/data/aquila2_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/data/aquila2_pretrain/FlagScale"
# 1B tokens for nnodes=1, model=7B
TRAINING_TOKENS = 1000000000

# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = 8
PIPELINE_PARALLEL = 1

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 2
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# gradient_accu_steps is the same as flagscale aquila-7B(9)
GLOBAL_BATCHSIZE = 2048
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "313000000000000"

# =========================================================
# network
# =========================================================
net_cmd ="export HCCL_CONNECT_TIMEOUT=3600;export HCCL_EXEC_TIMEOUT=0;export GLOO_SOCKET_IFNAME=bond0;export HCCL_SOCKET_IFNAME=bond0"
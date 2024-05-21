# =========================================================
# data
# =========================================================
DATA_DIR = "/work/home/zhaoying1/data/wudao_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/work/home/zhaoying1/work/pr_code/FlagScale-release-v0.2"
# 1B tokens for nnodes=1, model=7B
TRAINING_TOKENS = 20000000000

# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = 4
PIPELINE_PARALLEL = 20

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 1
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
# gradient_accu_steps is the same as flagscale aquila-70B(44)
GLOBAL_BATCHSIZE = 960
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

# =========================================================
# network
# =========================================================

# =========================================================
# mpirun 
# =========================================================
NP = 160
HOSTFILE = "/work/home/zhaoying1/work/pr_code/FlagPerf-AI_platform/training/dcu/hosts"

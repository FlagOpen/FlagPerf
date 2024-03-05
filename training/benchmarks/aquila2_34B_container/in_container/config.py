# =========================================================
# data
# =========================================================
DATA_DIR = "/data/aquila2_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/data/aquila2_pretrain/FlagScale"
# 1B tokens for nnodes=1, model=7B
TRAINING_TOKENS = 100000000

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
# gradient_accu_steps is the same as flagscale aquila-34B(32)
GLOBAL_BATCHSIZE = 32
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 4096

# =========================================================
# network
# =========================================================

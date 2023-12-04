# =========================================================
# data
# =========================================================
DATA_DIR = "/raid/dataset/aquila2_pretrain"
DATASET = "pile_wikipedia_demo"
FLAGSCALE_HOME = "FlagScale"
TRAINING_TOKENS = 10000000

# =========================================================
# parallel
# =========================================================
TENSOR_PARALLEL = 8
PIPELINE_PARALLEL = 1

# =========================================================
# batch
# =========================================================
MICRO_BATCHSIZE = 1
# globalbs = microbs * gradient_accu_steps * (worldsize/tp/pp)
GLOBAL_BATCHSIZE = 16
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 2048

# =========================================================
# network
# =========================================================

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================
# data
# =========================================================
DATA_DIR = "/work/home/zhaoying1/data/wudao_pretrain"
DATASET = "wudao_pretrain_text_document"
FLAGSCALE_HOME = "/work/home/zhaoying1/work/pr_code/FlagScale-release-v0.2"
# 1B tokens for nnodes=1, model=7B
TRAINING_TOKENS = 40000000
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
GLOBAL_BATCHSIZE = 480
# 2k for aquila2-7B, 4k for aquila2-34B and 70B
SEQLENGTH = 2048

# =========================================================
# network
# =========================================================

# =========================================================
# mpirun 
# =========================================================
NP = 16
HOSTFILE = "/work/home/zhaoying1/work/pr_code/FlagPerf-AI_platform/training/dcu/hosts"

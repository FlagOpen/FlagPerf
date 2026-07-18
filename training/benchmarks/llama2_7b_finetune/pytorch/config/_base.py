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

# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = "nvidia"
data_dir: str = None
name: str = "llama2_7b_finetune"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters
# =========================================================
# data
# =========================================================
dataset = "samsum_dataset"
batch_size_training: int = 1
val_batch_size: int = 1

model_name: str = "llama2_7b_hf"
weight_dir = model_name
mmlu_dir = ""
output_dir: str = ""

# =========================================================
# MFU
# =========================================================
seq_length = 512
theory_flops = 1.56e14

# =========================================================
# lora
# =========================================================
use_peft: bool = True
peft_method: str = "lora"  # None , llama_adapter, prefix

# =========================================================
# train && evaluate
# =========================================================
target_MMLU: float = 0.4
num_epochs: int = 3
few_shots = 5
gradient_accumulation_steps: int = 1
num_workers_dataloader: int = 1
lr: float = 1e-4
weight_decay: float = 0.0
gamma: float = 0.85
num_freeze_layers: int = 1

do_train = True
use_fp16: bool = False
mixed_precision: bool = False
distributed: bool = False
enable_fsdp: bool = False
low_cpu_fsdp: bool = False
run_validation: bool = False
freeze_layers: bool = False
one_gpu: bool = True
save_model: bool = False
dist_checkpoint_root_folder: str = "PATH/to/save/FSDP/model"  # will be used if using FSDP
dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
save_optimizer: bool = False  # will be used if using FSDP
use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels

# =========================================================
# utils
# =========================================================
seed: int = 42
dist_backend: str = 'nccl'
num_workers: int = 4
device: str = None

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 100
print_freq: int = 100
n_device: int = 1

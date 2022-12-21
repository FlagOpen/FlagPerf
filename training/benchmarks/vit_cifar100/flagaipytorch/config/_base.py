fp16 = True
# =========================================================
# data
# =========================================================
data_dir: str = None
train_data: str = "train"
eval_data: str = "eval"
save_dir: str = None
num_checkpoints: int = 1

# =========================================================
# train && evaluate
# =========================================================
batch_size: int = 8

lr: float = 1e-5
weight_decay: float = 0.1
gradient_accumulation_steps: int = 1
warmup: float = 0.1

eval_interval: int = 1000
save_interval: int = 100000

clip_grad: float = 1.0
seed: int = 10483

max_epochs: int = 50

log_freq: int = 1

not_call_launch: bool = True
local_rank: int = -1

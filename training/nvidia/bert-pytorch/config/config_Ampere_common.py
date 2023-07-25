from config_common import *
from torch.cuda.amp import GradScaler
import os

grad_scaler = GradScaler(init_scale=float(os.getenv("INIT_LOSS_SCALE", 2**20)),
                         growth_interval=2000, enabled=True)

fp16 = True
ddp_type = "apex"
dist_backend = "nccl"

#train_batch_size = 56 if get_gpu_mem() > 75 else 27
train_batch_size = 24

fused_gelu_bias = True
fused_mha = True
unpad = True
unpad_fmha = False
dense_seq_output = True
exchange_padding = True

dwu_num_rs_pg = 1
dwu_num_ar_pg = 1
dwu_num_blocks = 1

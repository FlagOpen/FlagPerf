from config_common import *


fp16 = True
ddp_type = "apex"
dist_backend = "nccl"

train_batch_size = 56 if get_gpu_mem() > 75 else 27

fused_gelu_bias = True
fused_mha = True
unpad = True
unpad_fmha = False
dense_seq_output = True
exchange_padding = True

dwu_num_rs_pg = 1
dwu_num_ar_pg = 1
dwu_num_blocks = 1

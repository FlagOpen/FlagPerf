vendor = "mthreads"

train_batch_size = 20
eval_batch_size = train_batch_size
lr = 0.000005  # fp32/amp
#lr = 0.00005   # bf16

dist_backend = "mccl"

amp = True
fp16 = False

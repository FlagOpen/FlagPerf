vendor = "mthreads"

train_batch_size = 20
eval_batch_size = train_batch_size
lr = 5e-06 / 0.2 # fp32/amp
#lr = 5e-05 / 0.6  # bf16

dist_backend = "mccl"

amp = True
fp16 = False

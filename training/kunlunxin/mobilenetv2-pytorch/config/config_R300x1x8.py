from config_common import *

train_batch_size = 256
eval_batch_size = 256

lr: float = 0.045 * 8

gradient_accumulation_steps = 1

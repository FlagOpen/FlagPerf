from config_common import *
# Batching
num_concat_batches = 1  #Keep NUM_NODES x $NUM_GPUS x $NUM_CONCAT_BATCHES x $UPDATE_FREQ = 64
update_freq = 1  # This config is for 1 NODE, 8 GPUS, so NUM_CONCAT_BATCHES=4, UPDATE_FREQ=2
max_sample_size = 250000
min_sample_size = 32000
max_tokens = 1400000
eval_steps = 100
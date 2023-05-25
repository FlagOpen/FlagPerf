# perf
name: str = "wav2vec2"
vendor: str = None
dist_backend: str = "nccl"
ddp_type: str = "native"
gradient_accumulation_steps: int = 1
target_acc=0.018
do_train=True
device: str = None
n_device: int = 1
log_frequency=log_freq=1
local_rank: int = -1
seed: int = 1

#important set
resume=False #[bool], default False, if True, read last_checkpoint from ckpt saved path
ckpt=None   #[str,path], default None, if True, given ckpt path
no_save=True #[bool], default True, if True , do not save ckpt
save_frequency=1 

# IO
output_dir="results/pretrain_base" 
data_dir="/workspace/wav2vec2_Perf/wav2vec2_data/LibriSpeech"
train_subset="train-full-960"
valid_subset="dev-other"

# Batching
num_concat_batches=4  #Keep NUM_NODES x $NUM_GPUS x $NUM_CONCAT_BATCHES x $UPDATE_FREQ = 64
update_freq=2  # This config is for 1 NODE, 8 GPUS, so NUM_CONCAT_BATCHES=4, UPDATE_FREQ=2
max_sample_size=250000
min_sample_size=32000
max_tokens=1400000 

# Training
fp16=False
bf16=False
mode="pretrain"
max_update=400000
loss_weights=[0.1, 10.0]
lr=0.0005
keep_milestones=[100, 200, 300, 400]
bf16_disable_loss_scaler=False
epochs_this_job=0
train_batch_size=num_concat_batches
eval_batch_size=None
skip_invalid_size_inputs_valid_test=True
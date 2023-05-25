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

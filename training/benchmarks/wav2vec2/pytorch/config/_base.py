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

# Model
extractor_mode="default"
mask_prob=0.65 
final_dim=256 
latent_temp=[2.0, 0.5, 0.999995]
encoder_layerdrop=0.05 
dropout_input=0.1 
dropout_features=0.1 
dropout=0.1 
attention_dropout=0.1 
encoder_layers=12 
encoder_embed_dim=768 
encoder_ffn_embed_dim=3072 
encoder_attention_heads=12 
feature_grad_mult=0.1 
ema=0.0
optimizer="adam" 
clip_norm=25 
weight_decay=0.01 
lr_policy="poly" 
lr_poly_power=1.0 
warmup_updates=32000 
max_tokens_valid=1400000 
hourglass_transformer="[2,(8,4),2]"
fp32_pos_conv=False
conv_pos=128
conv_pos_groups=16
activation_dropout=0.0
activation_fn='gelu'
apply_mask=False
layer_norm_first=False
rotary_embeddings=False
mha='pyt'
fp32_transformer_layernorm=False
fp32_mha_softmax=False
hourglass_resample='naive'
target_glu=False
adam_betas=[0.9, 0.98]
adam_eps=1e-06
conv_feature_layers='[(512,10,5)]+[(512,3,2)]*4+[(512,2,2)]+[(512,2,2)]'
conv_bias=False
fp32_conv_norms=False
infonce=True
log_keys=["prob_perplexity", "code_perplexity", "temp"]
enable_padding=False
normalize=False
sample_rate=16000
num_batch_buckets=0
required_batch_size_multiple=8
num_workers=6
batch_size_valid=None
batch_size=None
initial_lr_scale=0.0
final_lr_scale=0.0
hold_updates=0
benchmark_epochs_num=3
cpu=False
use_spectrogram_features=False
spectrogram_feature_stacking=1
spectrogram_feature_subsampling=1
spectrogram_window_size=0.02
spectrogram_window_stride=0.01
spectrogram_n_filt=80
quantize_input=False
mask_selection="static"
mask_other=0
mask_length=10
no_mask_overlap=False
mask_min_space=1
mask_channel_prob=0.0
mask_channel_before=False
mask_channel_selection="static"
mask_channel_other=0
mask_channel_length=10
no_mask_channel_overlap=False
mask_channel_min_space=1
num_negatives=100
cross_sample_negatives=0
codebook_negatives=0
logit_temp=0.1
fp32_cosine_sim=False
quantize_targets=True
latent_dim=0
latent_vars=320
latent_groups=2
quantizer_depth=1
quantizer_factor=3
negatives_from_everywhere=False
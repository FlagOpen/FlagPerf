# resume
# skip_invalid_size_inputs_valid_test 
# infonce
# quantize_targets 
# mha="pyt" 



# IO
output_dir="results/pretrain_base" 
data="/workspace/wav2vec2/wav2vec2_data/LibriSpeech "
train_subset="train-full-960 "
valid_subset="dev-other"

# Batching
max_tokens=1400000 
num_concat_batches=8 
update_freq=1 
max_sample_size=250000
min_sample_size=32000
save_frequency=1 


# Training
max_update=400000 
loss_weights="0.1 10.0"
lr=0.0005 

# Model
mask_prob=0.65 
extractor_mode="default" 
final_dim=256 
latent_temp="2.0 0.5 0.999995"
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



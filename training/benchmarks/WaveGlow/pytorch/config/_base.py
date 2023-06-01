# necessary
name: str = "WaveGlow"
dist_backend = "nccl"
vendor: str = "nvidia"
# tar_val_loss = -5.72  #https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2
tar_val_loss = -1  #https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2

#Perf
do_train = True
local_rank = -1
log_freq = 1
output = "result/"
log_file = "nvlog.json"
gradient_accumulation_steps = 1

# training
epochs = 250
batch_size = 10

# device
device: str = None
n_device: int = 1
fp16 = False
data_dir = None
world_size = None

# random seed
seed: int = None

# model args
amp = True
epochs_per_checkpoint = 50
learning_rate = 1e-4
segment_length = 8000
weight_decay = 0
grad_clip_thresh = 65504.0
cudnn_benchmark = True
cudnn_enabled = True
anneal_steps = None
bench_class = ''

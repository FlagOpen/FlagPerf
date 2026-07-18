# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# necessary
name: str = "WaveGlow"
dist_backend = "nccl"
vendor: str = "nvidia"
target_val_loss = -5.72  #https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2

save_checkpoint = False

#Perf
do_train = True
local_rank = -1
log_freq = 1
output = "training/result/"
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

# Dataset to use, default: wn18rr
dataset: str = "wn18rr"
# Model Name
model: str = "compgcn"
# Score Function for Link prediction
score_func: str = "conve"
# Composition Operation to be used in CompGCN
opn: str = "ccorr"

# Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0
gpu: int = 0

# Number of epochs
max_epochs: int = 5000

# L2 Regularization for Optimizer
l2: float = 0.0

# Starting Learning Rate
lr: float = 0.001
# Batch size
batch_size: int = 1024
# Label Smoothing
lbl_smooth: float = 0.1

# Number of processes to construct batches
num_workers: int = 10
# Seed for randomization
seed: 41504
# Number of basis relation vectors to use
num_bases: int = -1
# Initial dimension size for entities and relations
init_dim: int = 100

# List of output size for each compGCN layer
layer_size: list = [200]

# Dropout to use in GCN Layer
dropout: float = 0.1

# List of dropout value after each compGCN layer
layer_dropout: list = [0.3]
"""ConvE specific hyperparameters"""
# ConvE: Hidden dropout
hid_drop: float = 0.3
# ConvE: Feature Dropout
feat_drop: float = 0.3

# ConvE: k_w
k_w: int = 10
# ConvE: k_h
k_h: int = 20
# ConvE: Number of filters in convolution
num_filt: int = 200
# ConvE: Kernel size to use
ker_sz: int = 7
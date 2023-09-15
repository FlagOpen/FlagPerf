
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
# Model name
model_name = 'transformer'

# Path to the pretrained model. If set, only the mask head will be trained
frozen_weights: str = None

# * Backbone settings
# Name of the convolutional backbone to use
backbone = 'resnet50'
# If true, we replace stride with dilation in the last convolutional block (DC5)
dilation = False
# Type of positional embedding to use on top of the image features
position_embedding = 'sine'

# * Transformer parameters
# Number of encoding layers in the transformer
enc_layers = 6
# Number of decoding layers in the transformer
dec_layers = 6
# Intermediate size of the feedforward layers in the transformer blocks
dim_feedforward = 2048
# Size of the embeddings (dimension of the transformer)
hidden_dim = 256
# Dropout applied in the transformer
dropout = 0.1
# Number of attention heads inside the transformer's attentions
nheads = 8
# Number of query slots
num_queries = 100
pre_norm = False

# * Segmentation
# Train segmentation head if the flag is provided
masks = False
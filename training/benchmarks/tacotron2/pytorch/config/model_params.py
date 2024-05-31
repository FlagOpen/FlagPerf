"""Audio parameters"""
# Maximum audiowave value
max_wav_value: float = 32768.0
# Sampling rate
sampling_rate: int = 22050
# Filter length
filter_length: int = 1024
# Hop (stride) length
hop_length: int = 256
# Window length
win_length: int = 1024
# Minimum mel frequency
mel_fmin: float = 0.0
# Maximum mel frequency
mel_fmax: float = 8000
"""Misc parameters"""
# Number of bins in mel-spectrograms
n_mel_channels: int = 80
# Use mask padding
mask_padding: bool = False
"""Symbols parameters"""
# Number of symbols in dictionary
n_symbols: int = 148
# Input embedding dimension
symbols_embedding_dim: int = 512
"""Encoder parameters"""
# Encoder kernel size
encoder_kernel_size: int = 5
# Number of encoder convolutions
encoder_n_convolutions: int = 3
# Encoder embedding dimension
encoder_embedding_dim: int = 512
"""Decoder parameters"""
# Number of frames processed per step
n_frames_per_step: int = 1
# Number of units in decoder LSTM
decoder_rnn_dim: int = 1024
# Number of ReLU units in prenet layers
prenet_dim: int = 256
# Maximum number of output mel spectrograms
max_decoder_steps: int = 2000
# Probability threshold for stop token
gate_threshold: float = 0.5
# Dropout probability for attention LSTM
p_attention_dropout: float = 0.1
# Dropout probability for decoder LSTM
p_decoder_dropout: float = 0.1
# Stop decoding once all samples are finished
decoder_no_early_stopping: bool = False
"""Mel-post processing network parameters"""
# Postnet embedding dimension
postnet_embedding_dim: int = 512
# Postnet kernel size
postnet_kernel_size: int = 5
# Number of postnet convolutions
postnet_n_convolutions: int = 5
"""Attention parameters"""
# Number of units in attention LSTM
attention_rnn_dim: int = 1024
# Dimension of attention hidden representation
attention_dim: int = 128
"""Attention location parameters"""
# Number of filters for location-sensitive attention
attention_location_n_filters: int = 32
# Kernel size for location-sensitive attention
attention_location_kernel_size: int = 31

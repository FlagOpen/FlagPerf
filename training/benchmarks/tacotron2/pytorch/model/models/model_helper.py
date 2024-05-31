import torch
from .model import Tacotron2


def get_model(model_config,
              cpu_run,
              uniform_initialize_bn_weight=False,
              forward_is_infer=False,
              jittable=False):
    """get Tacotron2 model"""
    model = None

    if forward_is_infer:
        class Tacotron2__forward_is_infer(Tacotron2):

            def forward(self, inputs, input_lengths):
                return self.infer(inputs, input_lengths)

        model = Tacotron2__forward_is_infer(**model_config)
    else:
        model = Tacotron2(**model_config)

    if uniform_initialize_bn_weight:
        init_bn(model)

    if not cpu_run:
        model = model.cuda()
    return model

def get_model_config(args):
    model_config = dict(
        # optimization
        mask_padding=args.mask_padding,
        # audio
        n_mel_channels=args.n_mel_channels,
        # symbols
        n_symbols=args.n_symbols,
        symbols_embedding_dim=args.symbols_embedding_dim,
        # encoder
        encoder_kernel_size=args.encoder_kernel_size,
        encoder_n_convolutions=args.encoder_n_convolutions,
        encoder_embedding_dim=args.encoder_embedding_dim,
        # attention
        attention_rnn_dim=args.attention_rnn_dim,
        attention_dim=args.attention_dim,
        # attention location
        attention_location_n_filters=args.attention_location_n_filters,
        attention_location_kernel_size=args.attention_location_kernel_size,
        # decoder
        n_frames_per_step=args.n_frames_per_step,
        decoder_rnn_dim=args.decoder_rnn_dim,
        prenet_dim=args.prenet_dim,
        max_decoder_steps=args.max_decoder_steps,
        gate_threshold=args.gate_threshold,
        p_attention_dropout=args.p_attention_dropout,
        p_decoder_dropout=args.p_decoder_dropout,
        # postnet
        postnet_embedding_dim=args.postnet_embedding_dim,
        postnet_kernel_size=args.postnet_kernel_size,
        postnet_n_convolutions=args.postnet_n_convolutions,
        decoder_no_early_stopping=args.decoder_no_early_stopping
    )
    return model_config

def init_bn(module):
    """init batch_norm"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)

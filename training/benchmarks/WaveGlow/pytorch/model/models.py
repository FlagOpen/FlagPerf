# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from .model import WaveGlow
import torch


def model_parser(model_name, parser, add_help=False):
    if model_name == 'WaveGlow':
        from waveglow.arg_parser import waveglow_parser
        return waveglow_parser(parser, add_help)
    else:
        raise NotImplementedError(model_name)


def batchnorm_to_float(module):
    """Converts batch norm to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)


def get_model(model_name,
              model_config,
              uniform_initialize_bn_weight=False,
              forward_is_infer=False,
              jittable=False):
    """ Code chooses a model based on name"""
    model = None
    if model_name == 'WaveGlow':
        model = WaveGlow(**model_config)
        if forward_is_infer:
            model.forward = model.infer
    else:
        raise NotImplementedError(model_name)

    if uniform_initialize_bn_weight:
        init_bn(model)

    return model


def get_model_config(model_name, args):
    """ Code chooses a model based on name"""
    if model_name == 'WaveGlow':
        model_config = dict(n_mel_channels=args.n_mel_channels,
                            n_flows=args.flows,
                            n_group=args.groups,
                            n_early_every=args.early_every,
                            n_early_size=args.early_size,
                            WN_config=dict(n_layers=args.wn_layers,
                                           kernel_size=args.wn_kernel_size,
                                           n_channels=args.wn_channels))
        return model_config
    else:
        raise NotImplementedError(model_name)

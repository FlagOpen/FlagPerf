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

import argparse

def waveglow_parser(parent, add_help=False):
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)
    
    # misc parameters
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    # glow parameters
    parser.add_argument('--flows', default=12, type=int,
                        help='Number of steps of flow')
    parser.add_argument('--groups', default=8, type=int,
                        help='Number of samples in a group processed by the steps of flow')
    parser.add_argument('--early-every', default=4, type=int,
                        help='Determines how often (i.e., after how many coupling layers) \
                        a number of channels (defined by --early-size parameter) are output\
                        to the loss function')
    parser.add_argument('--early-size', default=2, type=int,
                        help='Number of channels output to the loss function')
    parser.add_argument('--sigma', default=1.0, type=float,
                        help='Standard deviation used for sampling from Gaussian')
    parser.add_argument('--segment-length', default=4000, type=int,
                        help='Segment length (audio samples) processed per iteration')

    # wavenet parameters
    wavenet = parser.add_argument_group('WaveNet parameters')
    wavenet.add_argument('--wn-kernel-size', default=3, type=int,
                        help='Kernel size for dialted convolution in the affine coupling layer (WN)')
    wavenet.add_argument('--wn-channels', default=512, type=int,
                        help='Number of channels in WN')
    wavenet.add_argument('--wn-layers', default=8, type=int,
                        help='Number of layers in WN')

    return parser



def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('--anneal-factor', type=float, choices=[0.1, 0.3], default=0.1,
                        help='Factor for annealing learning rate')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs-per-checkpoint', type=int, default=50,
                          help='Number of epochs per checkpoint')
    training.add_argument('--checkpoint-path', type=str, default='',
                          help='Checkpoint path to resume training')
    training.add_argument('--resume-from-last', action='store_true',
                          help='Resumes training from the last checkpoint; uses the directory provided with \'--output\' option to search for the checkpoint \"checkpoint_<model_name>_last.pt\"')
    training.add_argument('--dynamic-loss-scaling', type=bool, default=True,
                          help='Enable dynamic loss scaling')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument(
        '--use-saved-learning-rate', default=False, type=bool)
    optimization.add_argument('--grad-clip', default=5.0, type=float,
                              help='Enables gradient clipping and sets maximum gradient norm value')

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--load-mel-from-disk', action='store_true',
                         help='Loads mel spectrograms from disk instead of computing them on the fly')
    dataset.add_argument('--training_files',
                         default='filelists/ljs_audio_text_train_filelist.txt',
                         type=str, help='Path to training filelist')
    dataset.add_argument('--validation-files',
                         default='filelists/ljs_audio_text_val_filelist.txt',
                         type=str, help='Path to validation filelist')
    dataset.add_argument('--text-cleaners', nargs='*',
                         default=['english_cleaners'], type=str,
                         help='Type of text cleaners for input text')

    # audio parameters
    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=22050, type=int,
                       help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')

    return parser

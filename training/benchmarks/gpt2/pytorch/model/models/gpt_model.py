# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import torch

import config
from model.models.module import MegatronModule
from model.losses.cross_entropy import cross_entropy
from model.models.enums import AttnMaskType
from model.models.language_model import get_language_model
from model.models.utils import init_method_normal,scaled_init_method_normal
import math


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def post_language_model_processing(lm_output, labels, logit_weights, fp16_lm_cross_entropy):
    # Output. Format [s b h]
    output = torch.nn.functional.linear(lm_output, logit_weights, None)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = cross_entropy(output, labels)
        else:
            loss = cross_entropy(output.float(), labels)
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super(GPTModel, self).__init__()

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(config.init_method_std),
            scaled_init_method=scaled_init_method_normal(config.init_method_std,
                                                         config.num_layers),
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings(init_method_normal)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, inference_params=None):

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params)

        return post_language_model_processing(
                lm_output, labels,
                self.word_embeddings_weight(),
                self.fp16_lm_cross_entropy)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)



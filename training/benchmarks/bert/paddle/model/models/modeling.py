# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import sys
from functools import reduce
from io import open
from operator import mul

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer
try:
    from paddle.incubate.nn import FusedTransformerEncoderLayer
except ImportError:
    FusedTransformerEncoderLayer = None

# from model.layers.activations import ACT2FN
# #from model.layers.embeddings import BertEmbeddings
# from train.driver.distributed import get_rank

logger = logging.getLogger(__name__)


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file,
                      str) or (sys.version_info[0] == 2 and isinstance(
                          vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r",
                      encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):

        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                past_key_values_length=None):
        """
        call
        """
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            if past_key_values_length is not None:
                position_ids += past_key_values_length
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(Layer):
    """
    Pool the result of BertEncoder.
    """

    def __init__(self, hidden_size, pool_act="tanh"):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(Layer):
    """
    The bare BERT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `BertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `BertModel`.
            Defaults to `30522`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`.
            Defaults to `16`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `BertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

        pooled_act (str, optional):
            The non-linear activation function in the pooling layer.
            Defaults to `"tanh"`.
    """

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(mean=0.0,
                                         std=self.initializer_range if hasattr(
                                             self, "initializer_range") else
                                         self.bert.config["initializer_range"],
                                         shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12

    def __init__(self,
                 config,
                 initializer_range=0.02,
                 pad_token_id=0,
                 pool_act="tanh",
                 fuse=False):

        super(BertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(config)
        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            attn_dropout=config.attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             config.num_hidden_layers)
        self.pooler = BertPooler(config.hidden_size, pool_act)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r'''
        The BertModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import BertModel, BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
                model = BertModel.from_pretrained('bert-wwm-chinese')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''

        past_key_values_length = None
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros(
                    [batch_size, 1, 1, past_key_values_length],
                    dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask],
                                               axis=-1)

        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(
                    paddle.get_default_dtype())
                attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=past_key_values_length)
        self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
            cache=past_key_values,
        )
        # output_attentions=output_attentions,
        # output_hidden_states=output_hidden_states,
        # return_dict=return_dict
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            pooled_output = self.pooler(sequence_output)
            return (sequence_output, pooled_output)
        else:
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            # return BaseModelOutputWithPoolingAndCrossAttentions(
            #     last_hidden_state=sequence_output,
            #     pooler_output=pooled_output,
            #     past_key_values=encoder_outputs.past_key_values,
            #     hidden_states=encoder_outputs.hidden_states,
            #     attentions=encoder_outputs.attentions)


class BertPreTrainedModel(Layer):
    """
    An abstract class for pretrained BERT models. It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

        # we want to make sure vocab size is padded to % 8 == 0
        if self.config.vocab_size % 8 != 0:
            self.config.vocab_size += 8 - (self.config.vocab_size % 8)
            # if get_rank == 0:
            #     print(f'Padded vocab_size to : {self.config.vocab_size}')

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range if hasattr(
                            self, "initializer_range") else
                        self.bert_model_segment.bert.initializer_range,
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class BertLMPredictionHead(Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, config, embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = getattr(nn.functional, config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[config.vocab_size, config.hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[config.vocab_size],
            dtype=self.decoder_weight.dtype,
            is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            masked_positions = paddle.nonzero(masked_positions.reshape([-1]))
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class BertPretrainingHeads(Layer):
    """
    Perform language modeling task and next sentence classification task.

    Args:
        hidden_size (int):
            See :class:`BertModel`.
        vocab_size (int):
            See :class:`BertModel`.
        activation (str):
            Activation function used in the language modeling task.
        embedding_weights (Tensor, optional):
            Decoding weights used to map hidden_states to logits of the masked token prediction.
            Its data type should be float32 and its shape is [vocab_size, hidden_size].
            Defaults to `None`, which means use the same weights of the embedding layer.

    """

    def __init__(self, config, embedding_weights=None):
        super(BertPretrainingHeads, self).__init__()

        self.predictions = BertLMPredictionHead(config, embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        """
        Args:
            sequence_output(Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
            pooled_output(Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
            masked_positions(Tensor, optional):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


#这个东西似乎就是一个bert_model而已，为了对齐，所以先加上了
class BertForPreTrainingModelOnly(Layer):

    def __init__(self, config):
        super(BertForPreTrainingModelOnly, self).__init__()
        self.bert = BertModel(config)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                checkpoint_activations=False):
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
        )
        return sequence_output, pooled_output


class BertForPreTrainingHeadsOnly(Layer):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertForPreTrainingHeadsOnly, self).__init__()
        self.cls = BertPretrainingHeads(config, bert_model_embedding_weights)
        self.config = config

    def forward(self,
                sequence_output,
                pooled_output,
                labels=None,
                next_sentence_label=None,
                return_dict=False):
        # if dense_seq_output, prediction scores returned by this function is already masked out with masked_lm_labels, and first dimension is flattened
        # with paddle.static.amp.fp16_guard():
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output,
            labels)  #masked_positions这个参数应该是找到那些被mask的值

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            masked_lm_labels_flat = labels.reshape([-1])
            masked_lm_labels_dense = masked_lm_labels_flat[
                masked_lm_labels_flat != 0]
            masked_lm_loss = loss_fct(
                prediction_scores.reshape([-1, self.config.vocab_size]),
                masked_lm_labels_dense)
            # masked_lm_loss = loss_fct(
            #     prediction_scores.reshape(
            #         (-1, prediction_scores.shape[-1])),
            #     labels.reshape((-1, )))
            next_sentence_loss = loss_fct(
                seq_relationship_score.reshape((-1, 2)),
                next_sentence_label.reshape((-1, )))
            total_loss = masked_lm_loss + next_sentence_loss

        valid_mask = (masked_lm_labels_flat != 0)
        num_valid = valid_mask.sum(dtype='int64')
        mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != 0]
        mlm_predictions = prediction_scores.argmax(axis=-1).reshape([-1])
        mlm_acc = (mlm_predictions
                   == mlm_labels).sum(dtype='int64') / num_valid
        return total_loss, mlm_acc, num_valid

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss, ) +
                    output) if total_loss is not None else output


class BertForPreTraining(BertPreTrainedModel):
    """
    Bert Model with pretraining tasks on top.

    Args:
        bert (:class:`BertModel`):
            An instance of :class:`BertModel`.

    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert_model_segment = BertForPreTrainingModelOnly(config)
        self.heads_only_segment = BertForPreTrainingHeadsOnly(
            config,
            self.bert_model_segment.bert.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                next_sentence_label=None,
                masked_positions=None,
                position_ids=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids (Tensor, optional):
                See :class:`BertModel`.
            attention_mask (Tensor, optional):
                See :class:`BertModel`.
            masked_positions(Tensor, optional):
                See :class:`BertPretrainingHeads`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., vocab_size]`.
            next_sentence_label (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.bert.BertForPreTrainingOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.bert.BertForPreTrainingOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.bert.BertForPreTrainingOutput`.

        """
        sequence_output, pooled_output = self.bert_model_segment(
            input_ids, token_type_ids, attention_mask)
        return self.heads_only_segment(sequence_output, pooled_output, labels,
                                       next_sentence_label, return_dict)
        # return BertForPreTrainingOutput(
        #     loss=total_loss,
        #     prediction_logits=prediction_scores,
        #     seq_relationship_logits=seq_relationship_score,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

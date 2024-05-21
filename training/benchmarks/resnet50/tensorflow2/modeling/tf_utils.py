# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Common TF utilities."""

import functools
import six
import tensorflow as tf

from tensorflow.python.util import deprecation
from modeling import activations


@deprecation.deprecated(
    None,
    "tf.keras.layers.Layer supports multiple positional args and kwargs as "
    "input tensors. pack/unpack inputs to override __call__ is no longer "
    "needed.")
def pack_inputs(inputs):
    """Pack a list of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is None, replace it with a special constant
    tensor.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if x is None:
            outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
        else:
            outputs.append(x)
    return tuple(outputs)


@deprecation.deprecated(
    None,
    "tf.keras.layers.Layer supports multiple positional args and kwargs as "
    "input tensors. pack/unpack inputs to override __call__ is no longer "
    "needed.")
def unpack_inputs(inputs):
    """unpack a tuple of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is a special constant tensor, replace it
    with None.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if is_special_none_tensor(x):
            outputs.append(None)
        else:
            outputs.append(x)
    x = tuple(outputs)

    # To trick the very pointless 'unbalanced-tuple-unpacking' pylint check
    # from triggering.
    if len(x) == 1:
        return x[0]
    return tuple(outputs)


def is_special_none_tensor(tensor):
    """Checks if a tensor is a special None Tensor."""
    return tensor.shape.ndims == 0 and tensor.dtype == tf.int32


def get_activation(identifier, use_keras_layer=False, **kwargs):
    """Maps an identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

  It checks string first and if it is one of customized activation not in TF,
  the corresponding activation will be returned. For non-customized activation
  names and callable identifiers, always fallback to tf.keras.activations.get.

  Prefers using keras layers when use_keras_layer=True. Now it only supports
  'relu', 'linear', 'identity', 'swish', 'mish', 'leaky_relu', and 'gelu'.

  Args:
    identifier: String name of the activation function or callable.
    use_keras_layer: If True, use keras layer if identifier is allow-listed.
    **kwargs: Keyword arguments to use to instantiate an activation function.
      Available only for 'leaky_relu' and 'gelu' when using keras layers.
      For example: get_activation('leaky_relu', use_keras_layer=True, alpha=0.1)

  Returns:
    A Python function corresponding to the activation function or a keras
    activation layer when use_keras_layer=True.
  """
    if isinstance(identifier, six.string_types):
        identifier = str(identifier).lower()
        if use_keras_layer:
            keras_layer_allowlist = {
                "relu": "relu",
                "linear": "linear",
                "identity": "linear",
                "swish": "swish",
                "sigmoid": "sigmoid",
                "relu6": tf.nn.relu6,
                "leaky_relu": functools.partial(tf.nn.leaky_relu, **kwargs),
                "hard_swish": activations.hard_swish,
                "hard_sigmoid": activations.hard_sigmoid,
                "mish": activations.mish,
                "gelu": functools.partial(tf.nn.gelu, **kwargs),
            }
            if identifier in keras_layer_allowlist:
                return tf.keras.layers.Activation(
                    keras_layer_allowlist[identifier])
        name_to_fn = {
            "gelu": activations.gelu,
            "simple_swish": activations.simple_swish,
            "hard_swish": activations.hard_swish,
            "relu6": activations.relu6,
            "hard_sigmoid": activations.hard_sigmoid,
            "identity": activations.identity,
            "mish": activations.mish,
        }
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
            "equal to the expected tensor rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank)))


def safe_mean(losses):
    """Computes a safe mean of the losses.

  Args:
    losses: `Tensor` whose elements contain individual loss measurements.

  Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
    total = tf.reduce_sum(losses)
    num_elements = tf.cast(tf.size(losses), dtype=losses.dtype)
    return tf.math.divide_no_nan(total, num_elements)


def get_replica_id():
    """Gets replica id depending on the environment."""
    context = tf.distribute.get_replica_context()
    if context is not None:
        return context.replica_id_in_sync_group
    else:
        raise RuntimeError(
            "Unknown replica context. The `get_replica_id` method "
            "relies on TF 2.x tf.distribute API.")


def cross_replica_concat(value, axis, name="cross_replica_concat"):
    """Concatenates the given `value` across (GPU/TPU) cores, along `axis`.

  In general, each core ("replica") will pass a
  replica-specific value as `value` (corresponding to some element of a
  data-parallel computation taking place across replicas).

  The resulting concatenated `Tensor` will have the same shape as `value` for
  all dimensions except `axis`, where it will be larger by a factor of the
  number of replicas. It will also have the same `dtype` as `value`.

  The position of a given replica's `value` within the resulting concatenation
  is determined by that replica's replica ID. For
  example:

  With `value` for replica 0 given as

      0 0 0
      0 0 0

  and `value` for replica 1 given as

      1 1 1
      1 1 1

  the resulting concatenation along axis 0 will be

      0 0 0
      0 0 0
      1 1 1
      1 1 1

  and this result will be identical across all replicas.

  Note that this API only works in TF2 with `tf.distribute`.

  Args:
    value: The `Tensor` to concatenate across replicas. Each replica will have a
      different value for this `Tensor`, and these replica-specific values will
      be concatenated.
    axis: The axis along which to perform the concatenation as a Python integer
      (not a `Tensor`). E.g., `axis=0` to concatenate along the batch dimension.
    name: A name for the operation (used to create a name scope).

  Returns:
    The result of concatenating `value` along `axis` across replicas.

  Raises:
    RuntimeError: when the batch (0-th) dimension is None.
  """
    with tf.name_scope(name):
        context = tf.distribute.get_replica_context()
        # Typically this could be hit only if the tensor is derived from a
        # dataset with finite epochs and drop_remainder=False, where the last
        # batch could of different batch size and then the dim-0 is of dynamic
        # shape.
        if value.shape.as_list()[0] is None:
            raise RuntimeError(f"{value} has unknown batch.")
        return context.all_gather(value, axis=axis)


def clone_initializer(initializer):
    # Keras initializer is going to be stateless, which mean reusing the same
    # initializer will produce same init value when the shapes are the same.
    if isinstance(initializer, tf.keras.initializers.Initializer):
        return initializer.__class__.from_config(initializer.get_config())
    # When the input is string/dict or other serialized configs, caller will
    # create a new keras Initializer instance based on that, and we don't need to
    # do anything
    return initializer

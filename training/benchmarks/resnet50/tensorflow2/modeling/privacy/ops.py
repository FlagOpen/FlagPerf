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

"""Ops for differential privacy (gradient) transforms."""

from typing import List, Tuple
import tensorflow as tf


def clip_l2_norm(grads_vars: List[Tuple[tf.Tensor, tf.Tensor]],
                 l2_norm_clip: float) -> List[Tuple[tf.Tensor, tf.Tensor]]:
  """Clip gradients by global norm."""

  gradients = []
  variables = []
  for (g, v) in grads_vars:
    gradients.append(g)
    variables.append(v)
  clipped_gradients = tf.clip_by_global_norm(gradients, l2_norm_clip)[0]
  return list(zip(clipped_gradients, variables))


def add_noise(grads_vars: List[Tuple[tf.Tensor, tf.Tensor]],
              noise_stddev: float) -> List[Tuple[tf.Tensor, tf.Tensor]]:
  """Add noise to gradients."""
  ret = []
  for (g, v) in grads_vars:
    noise = tf.random.normal(tf.shape(g), stddev=noise_stddev)
    ret.append((g + noise, v))
  return ret


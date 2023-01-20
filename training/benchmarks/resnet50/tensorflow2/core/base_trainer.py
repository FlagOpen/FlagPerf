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

"""Standard Trainer implementation.

The base trainer implements the Orbit `StandardTrainable` and
`StandardEvaluable` interfaces. Trainers inside this project should be
interchangable and independent on model architectures and tasks.
"""
import functools
from typing import Union, Optional
from absl import logging
import gin
import orbit
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions
from official.modeling import optimization

ExperimentConfig = config_definitions.ExperimentConfig
TrainerConfig = config_definitions.TrainerConfig


class _AsyncTrainer(orbit.StandardTrainer, orbit.StandardEvaluator):
  """Trainer class for both sync and async Strategy."""

  def init_async(self):
    """Initializes the Async Trainer base class."""
    assert isinstance(self._strategy, tf.distribute.Strategy)
    self._is_async = isinstance(
        self._strategy, tf.distribute.experimental.ParameterServerStrategy)
    self._coordinator = None
    if self._is_async:
      self._coordinator = (
          tf.distribute.experimental.coordinator.ClusterCoordinator(
              self._strategy))

  def join(self):
    """Join all async steps. Only useful in aysnc training."""
    if getattr(self, "_is_async", False):
      self._coordinator.join()

  def create_train_loop_fn(self):
    """Creates a eval loop from the given step function and options."""
    train_loop_fn = super().create_train_loop_fn()
    if getattr(self, "_is_async", False):

      def _async_loop_fn(iterator, num_steps):
        self._coordinator.schedule(train_loop_fn, args=(iterator, num_steps))

      return _async_loop_fn
    else:
      return train_loop_fn

  def create_eval_loop_fn(self, has_state: bool):
    """Creates a training loop from the given step function and options."""
    eval_loop_fn = super().create_eval_loop_fn(has_state)

    if getattr(self, "_is_async", False):
      if has_state:
        raise ValueError(
            "Stateful eval loop is not supported in async training.")

      def _async_loop_fn(iterator, num_steps, state=None, reduce_fn=None):
        assert state is None
        assert reduce_fn is None
        self._coordinator.schedule(eval_loop_fn, args=(iterator, num_steps))

      return _async_loop_fn
    else:
      return eval_loop_fn

  def distribute_dataset(self, dataset_or_fn, *args, **kwargs):
    """A utility function to help create a `tf.distribute.DistributedDataset`.

    Args:
      dataset_or_fn: A instance of `tf.data.Dataset`, or a "dataset function"
        returning a `tf.data.Dataset`. If it is a function, it may optionally
        have an argument named `input_context` which will be passed a
        `tf.distribute.InputContext` instance.
      *args: Any positional arguments to pass through to `dataset_or_fn`.
      **kwargs: Any keyword arguments to pass through to `dataset_or_fn`.

    Returns:
      A distributed Dataset.
    """
    if getattr(self, "_is_async", False):
      per_worker_dataset_fn = functools.partial(
          orbit.utils.make_distributed_dataset, self._strategy, dataset_or_fn,
          *args, **kwargs)
      per_worker_dataset_fn = tf.function(per_worker_dataset_fn)

      return self._coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    else:
      return orbit.utils.make_distributed_dataset(self._strategy, dataset_or_fn,
                                                  *args, **kwargs)


def get_runtime_options(config: ExperimentConfig):
  """Get tf.distribute.RunOptions from config."""
  xla_options = {}
  if config.runtime.tpu_enable_xla_dynamic_padder is not None:
    xla_options["enable_xla_dynamic_padder"] = (
        config.runtime.tpu_enable_xla_dynamic_padder)
  return tf.distribute.RunOptions(
      experimental_xla_options=tf.tpu.XLAOptions(**xla_options))


@gin.configurable
class Trainer(_AsyncTrainer):
  """Implements the common trainer shared for TensorFlow models."""

  # pylint: disable=super-init-not-called
  def __init__(
      self,
      config: ExperimentConfig,
      task: base_task.Task,
      model: tf.keras.Model,
      optimizer: tf.optimizers.Optimizer,
      train: bool = True,
      evaluate: bool = True,
      train_dataset: Optional[Union[tf.data.Dataset,
                                    tf.distribute.DistributedDataset]] = None,
      validation_dataset: Optional[Union[
          tf.data.Dataset, tf.distribute.DistributedDataset]] = None,
      checkpoint_exporter=None):
    """Initialize common trainer for TensorFlow models.

    Args:
      config: An `ExperimentConfig` instance specifying experiment config.
      task: A base_task.Task instance.
      model: The model instance, e.g. a tf.keras.Model instance.
      optimizer: tf.optimizers.Optimizer instance.
      train: bool, whether or not this trainer will be used for training.
        default to True.
      evaluate: bool, whether or not this trainer will be used for evaluation.
        default to True.
      train_dataset: a dataset object created for training. With tf.distribute,
        it needs to be a `DistributedDataset`.
      validation_dataset: a dataset object created for evaluation. With
        tf.distribute, it needs to be a `DistributedDataset`. The evaluator will
        create a dataset iterator for each eval round, so the dataset does not
        need to repeat.
      checkpoint_exporter: an object that has the `maybe_export_checkpoint`
        interface.
    """
    # Gets the current distribution strategy. If not inside any strategy scope,
    # it gets a single-replica no-op strategy.
    self._strategy = tf.distribute.get_strategy()
    self._validate_params(
        config,
        check_train_data=train_dataset is None,
        check_validation_data=validation_dataset is None)
    self._config = config
    self._task = task
    self._model = model
    self._optimizer = optimizer
    self._checkpoint_exporter = checkpoint_exporter
    self._recovery = None
    # Runtime options are only applied to train_step.
    # We use default for eval_step.
    self._runtime_options = get_runtime_options(config)

    # Creates a shadow copy of the weights to store weights moving average.
    if isinstance(self._optimizer, optimization.ExponentialMovingAverage
                 ) and not self._optimizer.has_shadow_copy:
      self._optimizer.shadow_copy(self._model)

    # global_step increases by 1 after each training iteration.
    # We should have global_step.numpy() == self.optimizer.iterations.numpy()
    # when there is only 1 optimizer.
    self._global_step = orbit.utils.create_global_step()
    if hasattr(self.model, "checkpoint_items"):
      checkpoint_items = self.model.checkpoint_items
    else:
      checkpoint_items = {}
    self._checkpoint = tf.train.Checkpoint(
        global_step=self.global_step,
        model=self.model,
        optimizer=self.optimizer,
        **checkpoint_items)

    self._train_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
    self._validation_loss = tf.keras.metrics.Mean(
        "validation_loss", dtype=tf.float32)
    model_metrics = model.metrics if hasattr(model, "metrics") else []

    self.init_async()

    if train:
      self._train_metrics = self.task.build_metrics(
          training=True) + model_metrics
      train_dataset = train_dataset or self.distribute_dataset(
          self.task.build_inputs, self.config.task.train_data)
      orbit.StandardTrainer.__init__(
          self,
          train_dataset,
          options=orbit.StandardTrainerOptions(
              use_tf_while_loop=config.trainer.train_tf_while_loop,
              use_tf_function=config.trainer.train_tf_function,
              use_tpu_summary_optimization=config.trainer.allow_tpu_summary))

    if evaluate:
      self._validation_metrics = self.task.build_metrics(
          training=False) + model_metrics
      validation_dataset = validation_dataset or self.distribute_dataset(
          self.task.build_inputs, self.config.task.validation_data)
      orbit.StandardEvaluator.__init__(
          self,
          validation_dataset,
          options=orbit.StandardEvaluatorOptions(
              use_tf_function=config.trainer.eval_tf_function,
              use_tf_while_loop=config.trainer.eval_tf_while_loop))

  def _validate_params(self,
                       config,
                       check_train_data=True,
                       check_validation_data=True):
    r"""Validates if the configuration object passed to the Trainer.

    The experiment configuration should be structured as:
    \trainer
    \task
      \train_data
      \validation_data

    Args:
      config: a namedtuple, dataclass, ConfigDict, etc.
      check_train_data: whether to check task.train_data field.
      check_validation_data: whether to check task.validation_data field.
    """
    if not hasattr(config, "trainer"):
      raise AttributeError("The trainer requires the configuration contains an"
                           " attribute `trainer`.")

    if not hasattr(config, "task"):
      raise AttributeError("The trainer requires the configuration contains an"
                           " attribute `task`.")

    if check_train_data and not hasattr(config.task, "train_data"):
      raise AttributeError("The trainer requires the configuration contains an"
                           " attribute `task.train_data`.")

    if check_validation_data and not hasattr(config.task, "validation_data"):
      raise AttributeError("The trainer requires the configuration contains an"
                           " attribute `task.validation_data`.")

  @property
  def strategy(self):
    return self._strategy

  @property
  def config(self):
    return self._config

  @property
  def task(self):
    return self._task

  @property
  def model(self):
    return self._model

  @property
  def optimizer(self):
    if hasattr(self, "_optimizer"):
      return self._optimizer
    else:
      return None

  @property
  def global_step(self):
    return self._global_step

  @property
  def train_loss(self):
    """Accesses the training loss metric object."""
    return self._train_loss

  @property
  def validation_loss(self):
    """Accesses the validation loss metric object."""
    return self._validation_loss

  @property
  def train_metrics(self):
    """Accesses all training metric objects."""
    return self._train_metrics

  @property
  def validation_metrics(self):
    """Accesses all validation metric metric objects."""
    return self._validation_metrics

  def initialize(self):
    """A callback function.

    This function will be called when no checkpoint found for the model.
    If there is a checkpoint, the checkpoint will be loaded and this function
    will not be called. Tasks may use this callback function to load a
    pretrained checkpoint, saved under a directory other than the model_dir.
    """
    self.task.initialize(self.model)

  @property
  def checkpoint(self):
    """Accesses the training checkpoint."""
    return self._checkpoint

  @property
  def checkpoint_exporter(self):
    """Accesses the checkpoint exporter."""
    return self._checkpoint_exporter

  def train_loop_end(self):
    """See base class."""
    self.join()
    logs = {}
    for metric in self.train_metrics + [self.train_loss]:
      logs[metric.name] = metric.result()
      metric.reset_states()
    if callable(self.optimizer.learning_rate):
      # Maybe a self-implemented optimizer does not have `optimizer.iterations`.
      # So just to be safe here.
      if hasattr(self.optimizer, "iterations"):
        logs["learning_rate"] = self.optimizer.learning_rate(
            self.optimizer.iterations)
      else:
        logs["learning_rate"] = self.optimizer.learning_rate(self.global_step)
    else:
      logs["learning_rate"] = self.optimizer.learning_rate
    return logs

  def train_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      if self.config.runtime.enable_xla and (self.config.runtime.num_gpus > 0):
        task_train_step = tf.function(self.task.train_step, jit_compile=True)
      else:
        task_train_step = self.task.train_step
      logs = task_train_step(
          inputs,
          model=self.model,
          optimizer=self.optimizer,
          metrics=self.train_metrics)
      self._train_loss.update_state(logs[self.task.loss])
      self.global_step.assign_add(1)

    self.strategy.run(
        step_fn, args=(next(iterator),), options=self._runtime_options)

  def eval_begin(self):
    """Sets up metrics."""
    for metric in self.validation_metrics + [self.validation_loss]:
      metric.reset_states()
    # Swaps weights to test on weights moving average.
    if self.optimizer and isinstance(self.optimizer,
                                     optimization.ExponentialMovingAverage):
      self.optimizer.swap_weights()

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      logs = self.task.validation_step(
          inputs, model=self.model, metrics=self.validation_metrics)
      if self.task.loss in logs:
        self._validation_loss.update_state(logs[self.task.loss])
      return logs

    distributed_outputs = self.strategy.run(step_fn, args=(next(iterator),))
    return tf.nest.map_structure(self.strategy.experimental_local_results,
                                 distributed_outputs)

  def eval_end(self, aggregated_logs=None):
    """Processes evaluation results."""
    self.join()
    logs = {}
    for metric in self.validation_metrics:
      logs[metric.name] = metric.result()
    if self.validation_loss.count.numpy() != 0:
      logs[self.validation_loss.name] = self.validation_loss.result()
    else:
      # `self.validation_loss` metric was not updated, because the validation
      # loss was not returned from the task's `validation_step` method.
      logging.info("The task did not report validation loss.")
    if aggregated_logs:
      metrics = self.task.reduce_aggregated_logs(
          aggregated_logs, global_step=self.global_step)
      logs.update(metrics)

    if self._checkpoint_exporter:
      self._checkpoint_exporter.maybe_export_checkpoint(
          self.checkpoint, logs, self.global_step.numpy())
      metric_name = self.config.trainer.best_checkpoint_eval_metric
      logs["best_" +
           metric_name] = self._checkpoint_exporter.best_ckpt_logs[metric_name]

    # Swaps back weights after testing when EMA is used.
    # This happens after best checkpoint export so that average weights used for
    # eval are exported instead of regular weights.
    if self.optimizer and isinstance(self.optimizer,
                                     optimization.ExponentialMovingAverage):
      self.optimizer.swap_weights()
    return logs

  def eval_reduce(self, state=None, step_outputs=None):
    return self.task.aggregate_logs(state, step_outputs)

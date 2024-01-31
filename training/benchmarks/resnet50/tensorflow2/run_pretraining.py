"""Runs an Image Classification model."""
# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# Modified some functions to support FlagPerf.
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import os
import pprint
import time
import sys

from typing import Any, Mapping, Optional, Text, Tuple

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import dist_tensorflow2, Driver, Event
import argparse
from core import callbacks as custom_callbacks
from core import dataset_factory
from configs import base_configs
from configs import configs
from resnet import common
from modeling import hyperparams
from modeling import performance
from core import trainer_adapter
from utils import hyperparams_flags
from utils.misc import keras_utils

logger = None

def get_dtype_map() -> Mapping[str, tf.dtypes.DType]:
    """Returns the mapping from dtype string representations to TF dtypes."""
    return {
        'float32': tf.float32,
        'bfloat16': tf.bfloat16,
        'float16': tf.float16,
        'fp32': tf.float32,
        'bf16': tf.bfloat16,
    }


def _get_metrics(one_hot: bool) -> Mapping[Text, Any]:
    """Get a dict of available metrics to track."""
    if one_hot:
        return {
            # (name, metric_fn)
            'acc':
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'accuracy':
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'top_1':
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'top_5':
            tf.keras.metrics.TopKCategoricalAccuracy(k=5,
                                                     name='top_5_accuracy'),
        }
    else:
        return {
            # (name, metric_fn)
            'acc':
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'top_1':
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'top_5':
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name='top_5_accuracy'),
        }


def get_image_size_from_model(
        params: base_configs.ExperimentConfig) -> Optional[int]:
    """If the given model has a preferred image size, return it."""
    return None


def _get_dataset_builders(params: base_configs.ExperimentConfig,
                          strategy: tf.distribute.Strategy,
                          one_hot: bool) -> Tuple[Any, Any]:
    """Create and return train and validation dataset builders."""
    if one_hot:
        logging.warning(
            'label_smoothing > 0, so datasets will be one hot encoded.')
    else:
        logging.warning(
            'label_smoothing not applied, so datasets will not be one '
            'hot encoded.')

    num_devices = strategy.num_replicas_in_sync if strategy else 1

    image_size = get_image_size_from_model(params)

    dataset_configs = [params.train_dataset, params.validation_dataset]
    builders = []
    for config in dataset_configs:
        if config is not None and config.has_data:
            builder = dataset_factory.DatasetBuilder(config,
                                                     image_size=image_size
                                                     or config.image_size,
                                                     num_devices=num_devices,
                                                     one_hot=one_hot)
        else:
            builder = None
        builders.append(builder)

    return builders


def check_must_envconfigs(params):
    must_configs = [
        "FLAGPERF_NPROC", "FLAGPERF_HOSTS", "FLAGPERF_NODE_RANK",
        "FLAGPERF_HOSTS_PORTS"
    ]
    for config_item in os.environ.keys():
        if config_item in must_configs:
            must_configs.remove(config_item)
    if len(must_configs) > 0:
        raise ValueError("misses some env var items: " +
                         ", ".join(must_configs))
    params.local_rank = int(os.environ['FLAGPERF_NODE_RANK'])
    params.runtime.num_gpus = int(os.environ["FLAGPERF_NPROC"])

    if params.runtime.distribution_strategy == 'multi_worker_mirrored':
        hosts = os.environ["FLAGPERF_HOSTS"].split(",")
        ports = os.environ["FLAGPERF_HOSTS_PORTS"].split(",")
        params.runtime.worker_hosts = ",".join(
            [hosts[i] + ":" + ports[0] for i in range(len(hosts))])
        params.runtime.task_index = int(os.environ['FLAGPERF_NODE_RANK'])

    return params

def _get_params_from_flags(flags_obj: flags.FlagValues):
    """Get ParamsDict from flags."""

    global logger
    pp = pprint.PrettyPrinter()
    params = configs.get_config(flags_obj, model='resnet', dataset='imagenet')
    logging.info('_get_params_from_flags Base params: %s', pp.pformat(params.as_dict()))

    driver = Driver(params, [])
    driver.setup_config(argparse.ArgumentParser("renset50"))
    driver.setup_modules(driver, globals(), locals())
    params = driver.config
    params = check_must_envconfigs(params)
    logger = driver.logger

    params.validate()
    # params.lock()

    logging.info('Final model parameters: %s', pp.pformat(params.as_dict))

    return params, driver


def resume_from_checkpoint(model: tf.keras.Model, model_dir: str,
                           train_steps: int) -> int:
    """Resumes from the latest checkpoint, if possible.

  Loads the model weights and optimizer settings from a checkpoint.
  This function should be used in case of preemption recovery.

  Args:
    model: The model whose weights should be restored.
    model_dir: The directory where model weights were saved.
    train_steps: The number of steps to train.

  Returns:
    The epoch of the latest checkpoint, or 0 if not restoring.

  """
    logging.info('Load from checkpoint is enabled.')
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    logging.info('latest_checkpoint: %s', latest_checkpoint)
    if not latest_checkpoint:
        logging.info('No checkpoint detected.')
        return 0

    logging.info('Checkpoint file %s found and restoring from '
                 'checkpoint', latest_checkpoint)
    model.load_weights(latest_checkpoint)
    initial_epoch = model.optimizer.iterations // train_steps
    logging.info('Completed loading from checkpoint.')
    logging.info('Resuming from epoch %d', initial_epoch)
    return int(initial_epoch)


def initialize(params: base_configs.ExperimentConfig,
               dataset_builder: dataset_factory.DatasetBuilder):
    """Initializes backend related initializations."""
    keras_utils.set_session_config(enable_xla=params.runtime.enable_xla)
    performance.set_mixed_precision_policy(dataset_builder.dtype)
    if tf.config.list_physical_devices('GPU'):
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'
    tf.keras.backend.set_image_data_format(data_format)
    if params.runtime.run_eagerly:
        # Enable eager execution to allow step-by-step debugging
        tf.config.experimental_run_functions_eagerly(True)
    if tf.config.list_physical_devices('GPU'):
        if params.runtime.gpu_thread_mode:
            keras_utils.set_gpu_thread_mode_and_count(
                per_gpu_thread_count=params.runtime.per_gpu_thread_count,
                gpu_thread_mode=params.runtime.gpu_thread_mode,
                num_gpus=params.runtime.num_gpus,
                datasets_num_private_threads=params.runtime.
                dataset_num_private_threads)  # pylint:disable=line-too-long
        if params.runtime.batchnorm_spatial_persistent:
            os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

def define_classifier_flags():
    """Defines common flags for image classification."""
    hyperparams_flags.initialize_common_flags()
    flags.DEFINE_string('data_dir',
                        default=None,
                        help='The location of the input data.')
    flags.DEFINE_string(
        'mode',
        default=None,
        help='Mode to run: `train`, `eval`, `train_and_eval` or `export`.')
    flags.DEFINE_bool(
        'run_eagerly',
        default=None,
        help='Use eager execution and disable autograph for debugging.')
    flags.DEFINE_string('model_type',
                        default=None,
                        help='The type of the model, e.g. EfficientNet, etc.')
    flags.DEFINE_string('dataset',
                        default=None,
                        help='The name of the dataset, e.g. ImageNet, etc.')
    flags.DEFINE_integer(
        'log_steps',
        default=100,
        help='The interval of steps between logging of batch level stats.')
    flags.DEFINE_string('extern_config_dir',
                        default=None,
                        help='The testcase config dir.')
    flags.DEFINE_string('extern_config_file',
                        default=None,
                        help='The testcase config file.')
    flags.DEFINE_bool('enable_extern_config',
                      default=False,
                      help='Sets to enable non-standard config parameters.')
    flags.DEFINE_string('extern_module_dir',
                        default=None,
                        help='The extern module dir.')
    flags.DEFINE_string('vendor',
                        default=None,
                        help='AI chip vendor')

def serialize_config(params: base_configs.ExperimentConfig, model_dir: str):
    """Serializes and saves the experiment config."""
    params_save_path = os.path.join(model_dir, 'params.yaml')
    logging.info('Saving experiment configuration to %s', params_save_path)
    tf.io.gfile.makedirs(model_dir)
    hyperparams.save_params_dict_to_yaml(params, params_save_path)


def train_and_eval(params: base_configs.ExperimentConfig,
                   strategy_override: tf.distribute.Strategy,
                   driver) -> Mapping[str, Any]:
    """Runs the train and eval path using compile/fit."""
    logging.info('Running train and eval.')

    driver.event(Event.INIT_START)
    init_start_time = logger.previous_log_time

    dist_tensorflow2.configure_cluster(params.runtime.worker_hosts,
                                       params.runtime.task_index)
    # Note: for TPUs, strategy and scope should be created before the dataset
    strategy = strategy_override or dist_tensorflow2.get_distribution_strategy(
        distribution_strategy=params.runtime.distribution_strategy,
        all_reduce_alg=params.runtime.all_reduce_alg,
        num_gpus=params.runtime.num_gpus,
        tpu_address=params.runtime.tpu)

    strategy_scope = dist_tensorflow2.get_strategy_scope(strategy)

    logging.info('Detected %d devices.',
                 strategy.num_replicas_in_sync if strategy else 1)

    label_smoothing = params.model.loss.label_smoothing
    one_hot = label_smoothing and label_smoothing > 0

    builders = _get_dataset_builders(params, strategy, one_hot)
    datasets = [
        builder.build(strategy) if builder else None for builder in builders
    ]

    # Unpack datasets and builders based on train/val/test splits
    train_builder, validation_builder = builders  # pylint: disable=unbalanced-tuple-unpacking
    train_dataset, validation_dataset = datasets

    train_epochs = params.train.epochs
    train_steps = params.train.steps or train_builder.num_steps
    validation_steps = params.evaluation.steps or validation_builder.num_steps

    initialize(params, train_builder)

    logging.info('Global batch size: %d', train_builder.global_batch_size)

    with strategy_scope:
        model = trainer_adapter.convert_model(params)
        learning_rate = trainer_adapter.convert_learning_rate(
            params, train_builder, train_epochs, train_steps)
        optimizer = trainer_adapter.create_optimizer(params, learning_rate,
                                                     model, train_builder)
        metrics_map = _get_metrics(one_hot)
        metrics = [metrics_map[metric] for metric in params.train.metrics]
        steps_per_loop = train_steps if params.train.set_epoch_loop else 1

        if one_hot:
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=params.model.loss.label_smoothing)
        else:
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer,
                      loss=loss_obj,
                      metrics=metrics,
                      steps_per_execution=steps_per_loop)

        initial_epoch = 0
        print("params.train.resume_checkpoint", params.train.resume_checkpoint)
        if hasattr(params.train, "resume_checkpoint") and params.train.resume_checkpoint:
            initial_epoch = resume_from_checkpoint(
                model=model,
                model_dir=params.model_ckpt_dir,
                train_steps=train_steps)
        callbacks = custom_callbacks.get_callbacks(
            model_checkpoint=params.train.callbacks.
            enable_checkpoint_and_export,
            include_tensorboard=params.train.callbacks.enable_tensorboard,
            time_history=params.train.callbacks.enable_time_history,
            track_lr=params.train.tensorboard.track_lr,
            write_model_weights=params.train.tensorboard.write_model_weights,
            initial_step=initial_epoch * train_steps,
            batch_size=train_builder.global_batch_size,
            log_steps=params.train.time_history.log_steps,
            target_accuracy=params.target_accuracy,
            model_dir=params.model_dir,
            backup_and_restore=params.train.callbacks.enable_backup_and_restore
        )
    serialize_config(params=params, model_dir=params.model_dir)

    if params.evaluation.skip_eval:
        validation_kwargs = {}
    else:
        validation_kwargs = {
            'validation_data': validation_dataset,
            'validation_steps': validation_steps,
            'validation_freq': params.evaluation.epochs_between_evals,
        }

    driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time
    params.init_time = (init_end_time - init_start_time) / 1e+3
    
    # train
    stats_training = dict()
    raw_train_start_time = time.time()
    driver.event(Event.TRAIN_START)
    # model.fit()一次性加载整个数据集到内存中，适用于数据集较小，不会导致内存溢出的情况。
    history = model.fit(train_dataset,
                        epochs=train_epochs,
                        steps_per_epoch=train_steps,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks,
                        verbose=2,
                        **validation_kwargs)
    stats_training['no_eval_time'] = time.time() - raw_train_start_time

    driver.event(Event.TRAIN_END)

    # evaluate
    validation_output = None
    if not params.evaluation.skip_eval:
        validation_output = model.evaluate(validation_dataset,
                                           steps=validation_steps,
                                           verbose=2)

    # TODO(dankondratyuk): eval and save final test accuracy
    stats = common.build_stats(history, validation_output, callbacks)
    params.raw_train_time = time.time() - raw_train_start_time
    stats['no_eval_time'] = stats_training['no_eval_time']
    stats['raw_train_time'] = params.raw_train_time
    stats['pure_compute_time'] = stats_training['no_eval_time']

    return stats, params


def export(params: base_configs.ExperimentConfig):
    """Runs the model export functionality."""
    logging.info('Exporting model.')
    model = trainer_adapter.convert_model(params)
    checkpoint = params.export.checkpoint
    if checkpoint is None:
        logging.info('No export checkpoint was provided. Using the latest '
                     'checkpoint from model_dir.')
        checkpoint = tf.train.latest_checkpoint(params.model_dir)

    model.load_weights(checkpoint)
    model.save(params.export.destination)


def run(flags_obj: flags.FlagValues,
        strategy_override: tf.distribute.Strategy = None) -> Mapping[str, Any]:
    """Runs Image Classification model using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.
    strategy_override: A `tf.distribute.Strategy` object to use for model.

  Returns:
    Dictionary of training/eval stats
  """

    params, driver = _get_params_from_flags(flags_obj)
    if params.mode == 'train_and_eval':
        return train_and_eval(params, strategy_override, driver)
    elif params.mode == 'export_only':
        export(params)
    else:
        raise ValueError('{} is not a valid mode.'.format(params.mode))


def main(_):
    now = time.time()
    stats, params = run(flags.FLAGS)
    if stats:
        logging.info('Run stats:\n%s', stats)

    e2e_time = time.time() - now
    if params.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "num_trained_samples": stats['num_trained_samples'],
            "train_time": stats['raw_train_time'],
            "train_no_eval_time": stats['no_eval_time'],
            "pure_training_computing_time": stats['pure_compute_time'],
            "throughput(ips)_raw": stats['num_trained_samples'] / stats['raw_train_time'],
            "throughput(ips)_no_eval": stats['num_trained_samples'] / stats['no_eval_time'],
            "throughput(ips)_pure_compute": stats['num_trained_samples'] / stats['pure_compute_time'],
            "converged": stats["converged"],
            "final_accuracy": stats["accuracy_top_1"],
            "final_loss": stats["eval_loss"],
            "raw_train_time": params.raw_train_time,
            "init_time": params.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_classifier_flags()
    app.run(main)

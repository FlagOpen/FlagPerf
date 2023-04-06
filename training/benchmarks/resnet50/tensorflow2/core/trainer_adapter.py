from typing import Mapping
import tensorflow as tf

from . import optimizer_factory
from modeling import performance
from resnet import resnet_model
from configs import base_configs


def get_models() -> Mapping[str, tf.keras.Model]:
    """Returns the mapping from model type name to Keras model."""
    return {
        'resnet': resnet_model.resnet50,
    }


def convert_model(params):
    model_params = params.model.model_params.as_dict()
    model = get_models()[params.model.name](**model_params)
    return model


def convert_learning_rate(params, train_builder, train_epochs, train_steps):
    learning_rate = optimizer_factory.build_learning_rate(
        params=params.model.learning_rate,
        batch_size=train_builder.global_batch_size,
        train_epochs=train_epochs,
        train_steps=train_steps)
    return learning_rate


def create_optimizer(params, learning_rate, model, train_builder):
    use_fp16 = train_builder.dtype == 'float16'
    optimizer = optimizer_factory.build_optimizer(
        optimizer_name=params.model.optimizer.name,
        base_learning_rate=learning_rate,
        params=params.model.optimizer.as_dict(),
        model=model)
    optimizer = performance.configure_optimizer(
        optimizer, use_float16=use_fp16, loss_scale=get_loss_scale(params))
    return optimizer


def get_loss_scale(params: base_configs.ExperimentConfig,
                   fp16_default: float = 128.) -> float:
    """Returns the loss scale for initializations."""
    loss_scale = params.runtime.loss_scale
    if loss_scale == 'dynamic':
        return loss_scale
    elif loss_scale is not None:
        return float(loss_scale)
    elif (params.train_dataset.dtype == 'float32'
          or params.train_dataset.dtype == 'bfloat16'):
        return 1.
    else:
        assert params.train_dataset.dtype == 'float16'
        return fp16_default

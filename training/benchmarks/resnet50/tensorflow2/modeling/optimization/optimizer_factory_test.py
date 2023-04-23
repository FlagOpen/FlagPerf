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
"""Tests for optimizer_factory.py."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from .modeling.optimization import optimizer_factory
from .modeling.optimization.configs import optimization_config


class OptimizerFactoryTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(('sgd'), ('rmsprop'), ('adam'), ('adamw'),
                              ('lamb'), ('lars'), ('adagrad'))
    def test_optimizers(self, optimizer_type):
        params = {
            'optimizer': {
                'type': optimizer_type
            },
            'learning_rate': {
                'type': 'constant',
                'constant': {
                    'learning_rate': 0.1
                }
            }
        }
        optimizer_cls = optimizer_factory.LEGACY_OPTIMIZERS_CLS[optimizer_type]
        expected_optimizer_config = optimizer_cls().get_config()
        expected_optimizer_config['learning_rate'] = 0.1

        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()
        optimizer = opt_factory.build_optimizer(lr, postprocessor=lambda x: x)

        self.assertIsInstance(optimizer, optimizer_cls)
        self.assertEqual(expected_optimizer_config, optimizer.get_config())

    @parameterized.parameters(('sgd'), ('rmsprop'), ('adam'), ('adamw'),
                              ('lamb'), ('lars'), ('adagrad'))
    def test_new_optimizers(self, optimizer_type):
        params = {
            'optimizer': {
                'type': optimizer_type
            },
            'learning_rate': {
                'type': 'constant',
                'constant': {
                    'learning_rate': 0.1
                }
            }
        }
        optimizer_cls = optimizer_factory.NEW_OPTIMIZERS_CLS[optimizer_type]
        expected_optimizer_config = optimizer_cls().get_config()
        expected_optimizer_config['learning_rate'] = 0.1

        opt_config = optimization_config.OptimizationConfig(params)
        if optimizer_type == 'sgd':
            # Delete unsupported arg `decay` from SGDConfig.
            delattr(opt_config.optimizer.sgd, 'decay')
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()
        optimizer = opt_factory.build_optimizer(lr,
                                                postprocessor=lambda x: x,
                                                use_legacy_optimizer=False)

        self.assertIsInstance(optimizer, optimizer_cls)
        self.assertEqual(expected_optimizer_config, optimizer.get_config())

    def test_gradient_aggregator(self):
        params = {
            'optimizer': {
                'type': 'adam',
            },
            'learning_rate': {
                'type': 'constant',
                'constant': {
                    'learning_rate': 1.0
                }
            }
        }
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        # Dummy function to zero out gradients.
        zero_grads = lambda gv: [(tf.zeros_like(g), v) for g, v in gv]

        optimizer = opt_factory.build_optimizer(lr,
                                                gradient_aggregator=zero_grads)
        if isinstance(optimizer, tf.keras.optimizers.experimental.Optimizer):
            self.skipTest('New Keras optimizer does not support '
                          '`gradient_aggregator` arg.')

        var0 = tf.Variable([1.0, 2.0])
        var1 = tf.Variable([3.0, 4.0])

        grads0 = tf.constant([1.0, 1.0])
        grads1 = tf.constant([1.0, 1.0])

        grads_and_vars = list(zip([grads0, grads1], [var0, var1]))
        optimizer.apply_gradients(grads_and_vars)

        self.assertAllClose(np.array([1.0, 2.0]), var0.numpy())
        self.assertAllClose(np.array([3.0, 4.0]), var1.numpy())

    @parameterized.parameters((None, None), (1.0, None), (None, 1.0))
    def test_gradient_clipping(self, clipnorm, clipvalue):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'clipnorm': clipnorm,
                    'clipvalue': clipvalue
                }
            },
            'learning_rate': {
                'type': 'constant',
                'constant': {
                    'learning_rate': 1.0
                }
            }
        }

        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()
        optimizer = opt_factory.build_optimizer(lr)

        var0 = tf.Variable([1.0, 2.0])
        var1 = tf.Variable([3.0, 4.0])

        grads0 = tf.constant([0.1, 0.1])
        grads1 = tf.constant([2.0, 3.0])

        grads_and_vars = list(zip([grads0, grads1], [var0, var1]))
        optimizer.apply_gradients(grads_and_vars)

        self.assertAllClose(np.array([0.9, 1.9]), var0.numpy())
        if clipvalue is not None:
            self.assertAllClose(np.array([2.0, 3.0]), var1.numpy())
        elif clipnorm is not None:
            self.assertAllClose(np.array([2.4452999, 3.1679497]), var1.numpy())
        else:
            self.assertAllClose(np.array([1.0, 1.0]), var1.numpy())

    def test_missing_types(self):
        params = {'optimizer': {'type': 'sgd', 'sgd': {'momentum': 0.9}}}
        with self.assertRaises(ValueError):
            optimizer_factory.OptimizerFactory(
                optimization_config.OptimizationConfig(params))
        params = {
            'learning_rate': {
                'type': 'stepwise',
                'stepwise': {
                    'boundaries': [10000, 20000],
                    'values': [0.1, 0.01, 0.001]
                }
            }
        }
        with self.assertRaises(ValueError):
            optimizer_factory.OptimizerFactory(
                optimization_config.OptimizationConfig(params))

    def test_wrong_return_type(self):
        optimizer_type = 'sgd'
        params = {
            'optimizer': {
                'type': optimizer_type
            },
            'learning_rate': {
                'type': 'constant',
                'constant': {
                    'learning_rate': 0.1
                }
            }
        }

        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        with self.assertRaises(TypeError):
            _ = opt_factory.build_optimizer(0.1, postprocessor=lambda x: None)


# TODO(b/187559334) refactor lr_schedule tests into `lr_schedule_test.py`.

    def test_stepwise_lr_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'stepwise',
                'stepwise': {
                    'boundaries': [10000, 20000],
                    'values': [0.1, 0.01, 0.001]
                }
            }
        }
        expected_lr_step_values = [[0, 0.1], [5000, 0.1], [10000, 0.1],
                                   [10001, 0.01], [20000, 0.01],
                                   [20001, 0.001]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_stepwise_lr_with_warmup_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'stepwise',
                'stepwise': {
                    'boundaries': [10000, 20000],
                    'values': [0.1, 0.01, 0.001]
                }
            },
            'warmup': {
                'type': 'linear',
                'linear': {
                    'warmup_steps': 500,
                    'warmup_learning_rate': 0.01
                }
            }
        }
        expected_lr_step_values = [[0, 0.01], [250, 0.055], [500, 0.1],
                                   [5500, 0.1], [10000, 0.1], [10001, 0.01],
                                   [20000, 0.01], [20001, 0.001]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_exponential_lr_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'exponential',
                'exponential': {
                    'initial_learning_rate': 0.1,
                    'decay_steps': 1000,
                    'decay_rate': 0.96,
                    'staircase': True
                }
            }
        }
        expected_lr_step_values = [
            [0, 0.1],
            [999, 0.1],
            [1000, 0.096],
            [1999, 0.096],
            [2000, 0.09216],
        ]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_polynomial_lr_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'polynomial',
                'polynomial': {
                    'initial_learning_rate': 0.1,
                    'decay_steps': 1000,
                    'end_learning_rate': 0.001
                }
            }
        }

        expected_lr_step_values = [[0, 0.1], [500, 0.0505], [1000, 0.001]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_cosine_lr_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'cosine',
                'cosine': {
                    'initial_learning_rate': 0.1,
                    'decay_steps': 1000
                }
            }
        }
        expected_lr_step_values = [[0, 0.1], [250, 0.08535534],
                                   [500, 0.04999999], [750, 0.01464466],
                                   [1000, 0]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_constant_lr_with_warmup_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'constant',
                'constant': {
                    'learning_rate': 0.1
                }
            },
            'warmup': {
                'type': 'linear',
                'linear': {
                    'warmup_steps': 500,
                    'warmup_learning_rate': 0.01
                }
            }
        }

        expected_lr_step_values = [[0, 0.01], [250, 0.055], [500, 0.1],
                                   [5000, 0.1], [10000, 0.1], [20000, 0.1]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_stepwise_lr_with_polynomial_warmup_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'stepwise',
                'stepwise': {
                    'boundaries': [10000, 20000],
                    'values': [0.1, 0.01, 0.001]
                }
            },
            'warmup': {
                'type': 'polynomial',
                'polynomial': {
                    'warmup_steps': 500,
                    'power': 2.
                }
            }
        }
        expected_lr_step_values = [[0, 0.0], [250, 0.025], [500, 0.1],
                                   [5500, 0.1], [10000, 0.1], [10001, 0.01],
                                   [20000, 0.01], [20001, 0.001]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value, places=6)

    def test_power_lr_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'power',
                'power': {
                    'initial_learning_rate': 1.0,
                    'power': -1.0
                }
            }
        }
        expected_lr_step_values = [[0, 1.0], [1, 1.0], [250, 1. / 250.]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_power_linear_lr_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'power_linear',
                'power_linear': {
                    'initial_learning_rate': 1.0,
                    'power': -1.0,
                    'linear_decay_fraction': 0.5,
                    'total_decay_steps': 100,
                    'offset': 0,
                }
            }
        }
        expected_lr_step_values = [[0, 1.0], [1, 1.0], [40, 1. / 40.],
                                   [60, 1. / 60. * 0.8]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_power_with_offset_lr_schedule(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'power_with_offset',
                'power_with_offset': {
                    'initial_learning_rate': 1.0,
                    'power': -1.0,
                    'offset': 10,
                    'pre_offset_learning_rate': 3.0,
                }
            }
        }
        expected_lr_step_values = [[1, 3.0], [10, 3.0], [20, 1. / 10.]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)

    def test_step_cosine_lr_schedule_with_warmup(self):
        params = {
            'optimizer': {
                'type': 'sgd',
                'sgd': {
                    'momentum': 0.9
                }
            },
            'learning_rate': {
                'type': 'step_cosine_with_offset',
                'step_cosine_with_offset': {
                    'values': (0.0001, 0.00005),
                    'boundaries': (0, 500000),
                    'offset': 10000,
                }
            },
            'warmup': {
                'type': 'linear',
                'linear': {
                    'warmup_steps': 10000,
                    'warmup_learning_rate': 0.0
                }
            }
        }
        expected_lr_step_values = [[0, 0.0], [5000, 1e-4 / 2.0], [10000, 1e-4],
                                   [20000, 9.994863e-05], [499999, 5e-05]]
        opt_config = optimization_config.OptimizationConfig(params)
        opt_factory = optimizer_factory.OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()

        for step, value in expected_lr_step_values:
            self.assertAlmostEqual(lr(step).numpy(), value)


class OptimizerFactoryRegistryTest(tf.test.TestCase):

    def test_registry(self):

        class MyClass():
            pass

        optimizer_factory.register_optimizer_cls('test', MyClass)
        self.assertIn('test', optimizer_factory.LEGACY_OPTIMIZERS_CLS)
        with self.assertRaisesRegex(ValueError, 'test already registered.*'):
            optimizer_factory.register_optimizer_cls('test', MyClass)


if __name__ == '__main__':
    tf.test.main()

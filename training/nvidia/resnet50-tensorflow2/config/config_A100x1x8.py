# Training configuration for ResNet trained on ImageNet on GPUs.
# Reaches > 76.1% within 90 epochs.
# Note = This configuration uses a scaled per-replica batch size based on the number of devices.
# Base params  = base_configs.ExperimentConfig
do_train = True
model_dir = 'result'
model_ckpt_dir = ' '
mode = 'train_and_eval'
target_accuracy: float = 0.76
# runtime = dict(
#   distribution_strategy = 'multi_worker_mirrored',
#   run_eagerly = None,
#   tpu = None,
#   batchnorm_spatial_persistent = True)
runtime = dict(distribution_strategy='mirrored',
               run_eagerly=None,
               tpu=None,
               batchnorm_spatial_persistent=True)
train_dataset = dict(name='imagenet2012',
                     data_dir='/raid/dataset/ImageNet2012/tf_records',
                     builder='records',
                     split='train',
                     image_size=224,
                     num_classes=1000,
                     num_examples=1281167,
                     batch_size=128,
                     use_per_replica_batch_size=True,
                     dtype='float16',
                     mean_subtrat=True,
                     standardize=True)
validation_dataset = dict(
    name='imagenet2012',
    data_dir='/raid/dataset/ImageNet2012/tf_records',
    builder='records',
    split='validation',
    image_size=224,
    num_classes=1000,
    num_examples=50000,
    batch_size=128,  #256
    use_per_replica_batch_size=True,
    dtype='float16',
    mean_subtract=True,
    standardize=True)
model = dict(name='resnet',
             model_params=dict(rescale_inputs=False),
             optimizer=dict(name='momentum',
                            momentum=0.9,
                            decay=0.9,
                            epsilon=0.001),
             loss=dict(label_smoothing=0.1))
train = dict(resume_checkpoint=True,
             epochs=90,
             time_history=dict(log_steps=100),
             callbacks=dict(enable_checkpoint_and_export=False,
                            enable_backup_and_restore=False))
evaluation = dict(epochs_between_evals=1)

# local_rank for distributed training on gpus
local_rank: int = -1  ## for log

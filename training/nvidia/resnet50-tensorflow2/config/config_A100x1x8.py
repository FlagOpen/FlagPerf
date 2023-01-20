# Training configuration for ResNet trained on ImageNet on GPUs.
# Reaches > 76.1% within 90 epochs.
# Note = This configuration uses a scaled per-replica batch size based on the number of devices.
# Base params  = base_configs.ExperimentConfig
do_train = True
model_dir = 'result_ckpt'
mode = 'train_and_eval'
resume_from = '/mnt/data/ImageNet2012/resnet50_ckpt/'
# runtime =
#   distribution_strategy = 'multi_worker_mirrored'
#   num_gpus = 1  #8
#   worker_hosts= '10.1.2.155 =2222, 10.1.2.158:2223'
#   task_index: 0 
#   run_eagerly: None
#   tpu: None
#   batchnorm_spatial_persistent: true
runtime = dict(
  distribution_strategy = 'mirrored',
  num_gpus =  1 , #8
  run_eagerly =  None,
  tpu =  None,
  batchnorm_spatial_persistent = True)
train_dataset = dict(
  name = 'imagenet2012',
  data_dir = '/mnt/data/ImageNet2012/tf_records',
  # data_dir= '/raid/dataset/ImageNet2012/tf_records',
  builder = 'records',
  split = 'train',
  image_size = 224,
  num_classes = 1000,
  num_examples = 1281167,
  batch_size = 256,
  use_per_replica_batch_size = True,
  dtype = 'float16',
  mean_subtrat = True,
  standardize = True )
validation_dataset =  dict(
  name = 'imagenet2012',
  # data_dir = '/raid/dataset/ImageNet2012/tf_records/',
  data_dir = '/mnt/data/ImageNet2012/tf_records',
  builder = 'records',
  split = 'validation',
  image_size = 224,
  num_classes = 1000,
  num_examples = 50000,
  batch_size = 256,
  use_per_replica_batch_size = True,
  dtype = 'float16',
  mean_subtract = True,
  standardize = True)
model = dict(
  name = 'resnet',
  model_params = dict(
    rescale_inputs = False),
  optimizer = dict(
    name = 'momentum',
    momentum = 0.9,
    decay = 0.9,
    epsilon = 0.001),
  loss = dict(
    label_smoothing = 0.1))
train = dict(
  resume_checkpoint = True,
  epochs = 1,
  time_history = dict(
    log_steps = 100 ))
evaluation = dict(
  epochs_between_evals = 1)

# local_rank for distributed training on gpus
local_rank: int = -1 ##for log
# log_freq = 1
# train_batch_size = 8
# eval_batch_size = 8

# dist_backend = "nccl"

# lr = 1e-5
# weight_decay = 0.1
# adam_beta1 = 0.9
# adam_beta2 = 0.999
# adam_eps = 1e-08
# gradient_accumulation_steps = 1
# warmup = 0.1
# lr_decay_ratio = 0.1
# lr_decay_iters = 4338
# log_freq = 1
# seed = 10483
# max_samples_termination = 5553080
# training_event = None
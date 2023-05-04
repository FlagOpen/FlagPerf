from config_common import *


target_acc1 = 76.4

train_batch_size = 256
eval_batch_size = train_batch_size
max_steps = 1000000

learning_rate = 0.0001

beta_1: float = 0.9
beta_2: float = 0.99
eps: float = 1e-08

seed = 23333
<<<<<<< HEAD
max_samples_termination = 43912600
training_event = None
init_checkpoint = "checkpoint.70.pth.tar"
=======
max_samples_termination = 4391260000
>>>>>>> 6abcadc223c5a301eea089b6783092b46ba2aa10

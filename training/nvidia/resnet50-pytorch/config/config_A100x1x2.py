from config_common import *

train_batch_size = 256
eval_batch_size = train_batch_size
max_steps = 10000000
max_samples_termination = 4391260000

learning_rate = 0.001

beta_1: float = 0.9
beta_2: float = 0.99
eps: float = 1e-08

<<<<<<< HEAD
seed = 23333
training_event = None
init_checkpoint = "checkpoint.70.pth.tar"
=======
seed = 23333
>>>>>>> 6abcadc223c5a301eea089b6783092b46ba2aa10

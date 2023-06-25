from dataclasses import dataclass

@dataclass
class TrainingState:
    init_time = 0
    raw_train_time = 0

    global_steps = 0
    epoch: int = 1
    end_training: bool = False
    converged: bool = False

    loss: float = 0.0
    eval_MRR: float = 0.0
    eval_MR: float = 0.0
    eval_Hit1: float = 0.0
    eval_Hit3: float = 0.0
    eval_Hit10: float = 0.0

    def converged_success(self):
        self.end_training = True
        self.converged = True

from dataclasses import dataclass
import inspect
import torch


@dataclass
class TrainingState:
    
    def __init__(self) -> None:
        self.init_time: int = None
        self.raw_train_time: int = None
        self.global_steps: int = -1
        self.epoch:int = -1
        
        self.converged: bool = False

        # map
        self.P:float = 0
        self.R:float = 0
        self.mAP50:float = 0
        self.mAP:float = 0
        self.best_fitness:float = 0
        
        
    def status(self):
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        self.converged = True
        
    def _is_property(self, value):
        status = [
            not callable(value), not inspect.isclass(value),
            not inspect.ismodule(value), not inspect.ismethod(value),
            not inspect.isfunction(value), not inspect.isbuiltin(value),
            "classmethod object" not in str(value)
        ]
        return all(status)
        
    def to_dict(self, **kwargs):
        state_dict = dict()

        for var_name, value in self.__dict__.items():
            if not var_name.startswith("_") and self._is_property(value):
                state_dict[var_name] = value
        # todo  if get_last_lr() exist 
        lr = self._trainer.lr_scheduler.get_last_lr()
        if isinstance(lr, (tuple, list)):
            lr = lr[0]
        state_dict["learning_rate"] = lr
        # yolov5 
        exclude = [
            "eval_P", "eval_R", "eval_mAP_0_5", "eval_mAP_5_95", "skipped_steps",
            "converged", "init_time", "raw_train_time"
        ]
        for exkey in exclude:
            if exkey in state_dict:
                state_dict.pop(exkey)

        state_dict.update(kwargs)

        for k in state_dict.keys():
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].item()

        return state_dict
    
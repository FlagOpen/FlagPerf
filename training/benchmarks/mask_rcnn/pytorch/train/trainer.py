from torch.types import Device
import os

from model import create_model

from schedulers import create_scheduler
from optimizers import create_optimizer

import utils.train.train_eval_utils as utils
from train.evaluator import Evaluator
from train.training_state import TrainingState
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
from driver import Driver, Event, dist_pytorch


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None
        self.device = device
        self.optimizer = None
        self.config = config
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None

    def init(self):
        config = self.config
        pretrain_path = os.path.join(config.data_dir, config.pretrained_path)
        dist_pytorch.main_proc_print( f"backbone pretrain_path:{pretrain_path}" )

        self.model = create_model(self.config)
        self.model.to(self.device)
        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)
        # Attention: remember to move model to device before create_optimizer, otherwise, you will get a RuntimeError:
        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! when resuming training
        self.optimizer = create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        self.grad_scaler = self.adapter.create_grad_scaler()

    def train_one_epoch(self,
                        dataloader,
                        eval_dataloader,
                        print_freq=50,
                        warmup=True,
                        scaler=None):

        state = self.training_state
        driver = self.driver
        device = self.device
        model = self.model
        optimizer = self.optimizer
        driver.event(Event.EPOCH_BEGIN, state.epoch)

        mean_loss, lr = utils.train_one_epoch(model,
                                              self.adapter,
                                              optimizer,
                                              dataloader,
                                              device,
                                              state=state,
                                              print_freq=print_freq,
                                              warmup=warmup,
                                              scaler=scaler)
        dist_pytorch.main_proc_print(f"state.epoch: {state.epoch}")

        # update learning rate
        self.lr_scheduler.step()

        # evaluate after every epoch
        det_info, seg_info = utils.evaluate(model, eval_dataloader, device)

        if det_info is not None:
            state.eval_map_bbox = det_info[0]

        if seg_info is not None:
            state.eval_map_segm = seg_info[0]

        driver.event(Event.EPOCH_END, state.epoch)
        # check training state
        self.detect_training_status()
        return mean_loss, lr

    def detect_training_status(self):
        state = self.training_state
        config = self.config
        if state.eval_map_bbox >= config.target_map_bbox and state.eval_map_segm >= config.target_map_segm:
            dist_pytorch.main_proc_print(
                f"converged_success. eval_map_bbox: {state.eval_map_bbox}, eval_map_segm:{state.eval_map_segm} \
                    target_map_bbox: {config.target_map_bbox}. target_map_segm:{config.target_map_segm}")
            state.converged_success()

        return state.end_training
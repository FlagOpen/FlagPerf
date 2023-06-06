import torch
from torch.types import Device
import math
import time

from train.evaluator import Evaluator
from train.training_state import TrainingState

from driver import Driver, Event
from fairseq.ddp_trainer import DDPTrainer
from fairseq.models import build_model
from fairseq.data import data_utils


class Trainer(DDPTrainer):

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config):
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.scaler = None

        self.device = device
        self.optimizer = None
        self.config = config
        self.model = build_model(config).to(device)
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.ddp_trainer = None
        super(Trainer, self).__init__(self.config, self.model)
        # Send a dummy batch to warm the caching allocator
        src_dict, tgt_dict = data_utils.load_dictionaries(config)
        dummy_batch = data_utils.get_dummy_batch(config.max_tokens, src_dict, tgt_dict)
        self.dummy_train_step(dummy_batch)

    def train_one_epoch(self, dataloader):
        """Train the model for one epoch."""
        args = self.config
        epoch_itr = dataloader
        trainer = self
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)
        # Initialize data iterator
        itr = epoch_itr.next_epoch_itr()

        # update parameters every N batches
        if epoch_itr.epoch <= len(args.update_freq):
            update_freq = args.update_freq[epoch_itr.epoch - 1]
        else:
            update_freq = args.update_freq[-1]

        num_batches = len(epoch_itr)

        trainer.get_throughput_meter().reset()
        for i, sample in enumerate(itr):
            state.global_steps += 1
            update_params = not (i < num_batches - 1 and (i + 1) % update_freq > 0)
            if update_params:
                driver.event(Event.STEP_BEGIN, step=state.global_steps)
            trainer.train_step(sample, update_params=update_params, last_step=(i == len(itr)-1))
            if not update_params:
                continue
            state.lr = trainer.get_lr()
            state.loss = self.avg_loss_meter.avg
            other_state = {"tokens/s": self.throughput_meter.avg}
            driver.event(Event.STEP_END,
                         message=state.to_dict(**other_state),
                         step=state.global_steps,
                         loss=state.loss)
            # ignore the first mini-batch in words-per-second calculation
            if i == 0:
                trainer.get_throughput_meter().reset()

            end_training = self.detect_training_status(state)
            if end_training:
                break

        if epoch_itr.epoch % args.validate_interval == 0:
            eval_start = time.time()
            state.eval_loss = self.evaluator.evaluate(trainer)
            if state.eval_loss <= args.target_loss:
                state.converged_success()
            eval_end = time.time()
            eval_result = dict(global_steps=state.global_steps,
                               eval_loss=state.eval_loss,
                               time=eval_end - eval_start)
            driver.event(Event.EVALUATE, eval_result)

        trainer.lr_step(epoch_itr.epoch, state.eval_loss)
        torch.cuda.synchronize()
        driver.event(Event.EPOCH_END, state.epoch)

    def detect_training_status(self, state):
        config = self.config
        max_update = config.max_update or math.inf

        if state.end_training \
                or (self.get_num_updates() >= max_update) \
                or (not state.lr >= config.min_lr):
            state.end_training = True

        return state.end_training

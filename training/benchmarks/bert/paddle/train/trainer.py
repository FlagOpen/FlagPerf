import math
import time
from tabnanny import check
from collections import OrderedDict


import config
import torch
from model import create_model
from schedulers import create_scheduler
from torch.types import Device
from utils.checkpoint import remap_segmented_model_parameters

from train.driver import Event
from train.evaluator import Evaluator
from train.training_state import TrainingState

from .driver import Driver, distributed

import paddle

class Trainer():

    def __init__(self, driver: Driver, adapter,
                 evaluator: Evaluator,
                 training_state: TrainingState):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.grad_scaler = None
        self.optimizer = None
        self.bert_config = None
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.global_batch_size = None
        self.overflow_buf = None

    def init(self):
        self.bert_config, self.model = create_model(config)
        self.model = self._init_model(self.model)
        self.model = self.adapter.convert_model(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer)
        self.optimizer = self.adapter.create_optimizer(self.model,self.lr_scheduler)
        self.model, self.optimizer = self.adapter.model_to_fp16(
             self.model, self.optimizer)
        self.model = self.adapter.model_to_ddp(self.model)
        
        self.global_batch_size = distributed.global_batch_size(config)
        # self.grad_scaler = self.adapter.create_grad_scaler()

    def _init_model(self, model):
        def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path="pytorch_model.bin",
                                                paddle_dump_path="model_state.pdparams",
                                                version="old", ):
            hf_to_paddle = {
                "embeddings.LayerNorm": "embeddings.layer_norm",
                "encoder.layer": "encoder.layers",
                "attention.self.query": "self_attn.q_proj",
                "attention.self.key": "self_attn.k_proj",
                "attention.self.value": "self_attn.v_proj",
                "attention.output.dense": "self_attn.out_proj",
                "intermediate.dense": "linear1",
                "output.dense": "linear2",
                "attention.output.LayerNorm": "norm1",
                "output.LayerNorm": "norm2",
                "predictions.decoder.": "predictions.decoder_",
                "predictions.transform.dense": "predictions.transform",
                "predictions.transform.LayerNorm": "predictions.layer_norm",
            }
            do_not_transpose = []
            if version == "old":
                hf_to_paddle.update({
                    "predictions.bias": "predictions.decoder_bias",
                    ".gamma": ".weight",
                    ".beta": ".bias",
                })
                do_not_transpose = do_not_transpose + ["predictions.decoder.weight"]

            pytorch_state_dict = torch.load(
                pytorch_checkpoint_path, map_location="cpu")
            pytorch_state_dict = pytorch_state_dict['model']
            paddle_state_dict = OrderedDict()
            for k, v in pytorch_state_dict.items():
                is_transpose = False
                if k[-7:] == ".weight":
                    # embeddings.weight and LayerNorm.weight do not transpose
                    if all(d not in k for d in do_not_transpose):
                        if ".embeddings." not in k and ".LayerNorm." not in k:
                            if v.ndim == 2:
                                v = v.transpose(0, 1)
                                is_transpose = True
                oldk = k
                for hf_name, pd_name in hf_to_paddle.items():
                    k = k.replace(hf_name, pd_name)

                # add prefix `bert.`
                if "bert." not in k and "cls." not in k and "classifier" not in k:
                    k = "bert." + k

                #print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
                paddle_state_dict[k] = v.data.numpy()

            paddle.save(paddle_state_dict, paddle_dump_path)
            return paddle_state_dict,pytorch_state_dict

        checkpoint,_ = convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path=config.data_dir+'/model.ckpt-28252.pt',
                                                paddle_dump_path=config.data_dir+'/model_state.pdparams',
                                                version="old", )

        checkpoint_remapped = remap_segmented_model_parameters(checkpoint)

        model.load_dict(checkpoint_remapped)
        return model

    def train_one_epoch(self, dataloader):
        state = self.training_state
        driver = self.driver
        driver.event(Event.EPOCH_BEGIN, state.epoch)
        
        step_start_time = time.time()

        for dataloader_idx, batch_idx, batch in dataloader.iter_batchs():
            state.num_trained_samples = state.global_steps * self.global_batch_size

            state.global_steps += 1
            state.iter_dataloader_idx = dataloader_idx
            driver.event(Event.STEP_BEGIN, step=state.global_steps)
            train_loss,train_mlm_acc = self.train_one_step(batch_idx, batch)

            other_state = dict()
            if state.global_steps % config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time

                sequences_per_second = (distributed.global_batch_size(
                    config) * config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()
                state.eval_loss, state.eval_mlm_accuracy = self.evaluator.evaluate(
                    self)
                eval_end = time.time()
                state.eval_loss = state.eval_loss
                
                eval_result = dict(global_steps=state.global_steps,
                                   eval_loss=state.eval_loss,
                                   eval_mlm_accuracy=state.eval_mlm_accuracy,
                                   time=eval_end - eval_start)  # elapsed = eval_end - eval_start

            end_training = self.detect_training_status(state)
            other_state["eval_mlm_accuracy"] = state.eval_mlm_accuracy
            state_info = state.to_dict(**other_state)
            driver.event(Event.STEP_END, message=state_info, step=state.global_steps,
                         loss=state.loss)

            if eval_result is not None:
                driver.event(Event.EVALUATE, eval_result)

            if end_training:
                break

        driver.event(Event.EPOCH_END, state.epoch)

    def train_one_step(self, batch_idx, batch):

        state = self.training_state

        self.model.train()
        state.loss, state.mlm_acc, _ = self.forward(batch)
        self.adapter.backward(state.global_steps, state.loss,
                              self.optimizer, grad_scaler=self.grad_scaler)
        self.driver.event(Event.BACKWARD, state.global_steps,
                          state.loss, self.optimizer, self.grad_scaler)
        self.lr_scheduler.step()

        return state.loss.numpy(), state.mlm_acc.numpy()

    def detect_training_status(self, state: TrainingState):
        if state.eval_mlm_accuracy >= config.target_mlm_accuracy:
            state.converged_success()

        if state.global_steps > config.max_steps or state.num_trained_samples > config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state: TrainingState):
        do_eval = all([
            config.eval_dir is not None,
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps % math.ceil(
                config.eval_interval_samples / distributed.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])
 
        


        return do_eval or state.global_steps >= config.max_steps

    def forward(self, batch):
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        loss, mlm_acc, num_valid = self.model(input_ids, segment_ids, input_mask,
                                              masked_lm_labels, next_sentence_labels)
        return loss, mlm_acc, num_valid

    def inference(self, batch):
        self.model.eval()
        return self.forward(batch)

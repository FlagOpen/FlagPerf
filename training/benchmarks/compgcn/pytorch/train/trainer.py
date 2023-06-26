from time import time

import torch
import torch.utils.data
from torch.types import Device
import numpy as np

from dataloaders.dataloader import Data
from model import create_model
from optimizers import create_optimizer
from train.evaluator import Evaluator
from train.training_state import TrainingState
from driver import Driver, dist_pytorch
from utils.utils import in_out_norm


class Trainer:

    def __init__(self, driver: Driver, adapter, evaluator: Evaluator,
                 training_state: TrainingState, device: Device, config,
                 data: Data):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator

        self.data = data
        graph = data.g.to(device)
        self.graph = in_out_norm(graph)

    def init(self):
        device = torch.device(self.config.device)
        dist_pytorch.main_proc_print("Init progress:")
        self.model = create_model(self.config, self.data)
        self.model.to(self.device)

        self.model = self.adapter.convert_model(self.model)
        self.model = self.adapter.model_to_fp16(self.model)
        self.model = self.adapter.model_to_ddp(self.model)

        self.optimizer = create_optimizer(self.model, self.config)
        self.optimizer.zero_grad()

    def train_one_epoch(self, data):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        epoch = self.training_state.epoch
        config = self.config
        state = self.training_state

        data_iter = data.data_iter
        graph = self.graph

        best_mrr = 0.0
        kill_cnt = 0

        # loss function
        loss_fn = torch.nn.BCELoss()

        # Training and validation using a full graph
        model.train()
        train_loss = []
        t0 = time()

        for step, batch in enumerate(data_iter["train"]):
            state.global_steps += 1
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            
            logits = model(graph, sub, rel)

            # compute loss
            tr_loss = loss_fn(logits, label)
            train_loss.append(tr_loss.item())

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        train_loss = np.sum(train_loss)

        t1 = time()
        val_results = self.evaluator.evaluate(model,
                                              graph,
                                              device,
                                              data_iter,
                                              split="valid")
        state.eval_MRR = val_results['mrr']
        state.eval_Hit1, state.eval_Hit3, state.eval_Hit10 = val_results[
            'hits@1'], val_results['hits@3'], val_results['hits@10']
        t2 = time()

        # validate
        if val_results["mrr"] > best_mrr:
            best_mrr = val_results["mrr"]
            best_epoch = epoch
            torch.save(model.state_dict(), "comp_link" + "_" + config.dataset)
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt > 100:
                print("early stop.")
                return
        print(
            "In epoch {}, Train Loss: {:.4f}, Valid MRR: {:.5},  Valid Hits@1: {:.5}, Train time: {}, Valid time: {}"
            .format(epoch, train_loss, val_results["mrr"], state.eval_Hit1,
                    t1 - t0, t2 - t1))

        if state.eval_MRR >= config.target_MRR and state.eval_Hit1 >= config.target_Hit1:
            dist_pytorch.main_proc_print(f"converged_success")
            state.converged_success()

        if state.epoch > config.max_epochs:
            state.end_training = True

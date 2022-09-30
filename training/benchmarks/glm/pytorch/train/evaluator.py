# coding=utf-8
import torch
import datetime
import subprocess

import config
from utils import print_rank_0
#from train.trainer import process_batch


class Evaluator:
    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.args = args

    def evaluate(self, trainer):
        print_rank_0("calculating evaluate metrics ...")
        score = multichoice_evaluate(
            trainer.model, self.dataloader, self.args)
        return score

def multichoice_evaluate(model, dataloader, args, segment_length=10):
    model.eval()
    total_sample = torch.tensor(0, device=args.device)
    total_score = torch.tensor(0, device=args.device)

    with torch.no_grad():
        # For all the batches in the dataset.
        for batch_index, batch in enumerate(dataloader):
            data = {t: batch[t].to(args.device) for t in batch if t != 'answer_idx'}
            tokens, position_ids, attention_mask, target_ids, logit_mask = data[
                'text'], data['position'], data['mask'], data['target'], data['logit_mask']
            inputs = [tokens, position_ids,
                      attention_mask, target_ids, logit_mask]
            # if choice length max than 10
            if len(inputs[0].shape) == 3 and inputs[0].size(1) > segment_length:
                logit_list = []
                for i in range((inputs[0].size(1) - 1) // segment_length + 1):
                    input_batch = [
                        arg[:, i * segment_length: (i + 1) * segment_length] for arg in inputs]
                    logits, *mems = model(*input_batch)
                    logit_list.append(logits)
                logits = torch.cat(logit_list, dim=1)
            else:
                logits, *mems = model(*inputs)

            loss_mask = data["loss_mask"]
            logits = logits * loss_mask - 1000000000.0 * (1.0 - loss_mask)

            predicted = torch.argmax(logits, dim=-1).tolist()
            true_labels = batch['answer_idx']

            batch_score = em_evaluate(predicted, true_labels)

            total_sample += len(predicted)
            total_score += batch_score

    model.train()
    #config.training_event_instance.device_barrier()
    torch.distributed.all_reduce(
        total_sample, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(
        total_score, op=torch.distributed.ReduceOp.SUM)

    # print(f"samples:{total_sample}, score:{total_score}")
    score = total_score/total_sample

    return score.item()


def em_evaluate(predictions, labels):
    assert len(predictions) == len(labels)
    score = 0
    for pred, true_list in zip(predictions, labels):
        if pred in true_list:
            score += 1
    # score = 100.0 * score / len(predictions)
    return score

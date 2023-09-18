import os

import nltk
import numpy as np
import evaluate
import torch
import torch.distributed as dist


def postprocess_text(preds, labels):
    """
        https://github.com/huggingface/transformers/blob/v4.31.0/examples/pytorch/summarization/run_summarization.py#L621
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def pad_across_processes(config, preds, labels, tokenizer):
    if not config.distributed:
        return preds, labels

    max_pred_len = torch.tensor(preds.shape[1],
                                dtype=torch.int64,
                                device=config.device)
    dist.all_reduce(max_pred_len, dist.ReduceOp.MAX)
    max_pred_len = int(max_pred_len)

    if max_pred_len > preds.shape[1]:
        pad_index = tokenizer.pad_token_id
        new_preds = preds.new_zeros(preds.shape[0], max_pred_len) + pad_index
        new_preds[:, :preds.shape[1]] = preds
        preds = new_preds

    all_preds = [preds.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(all_preds, preds)

    all_labels = [labels.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(all_labels, labels)

    return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)


class Evaluator:
    """Evaluator"""
    def __init__(self, config):
        self.config = config
        nltk.data.path.append(os.path.join(config.data_dir, 'nltk_data'))
        self.metric_path = os.path.join(config.data_dir, 'metrics', 'rouge',
                                        'rouge.py')
        self.reset()

    def reset(self):
        self.metric = evaluate.load(self.metric_path)

    def add_batch(self, tokenizer, preds, labels):
        preds, labels = pad_across_processes(self.config, preds, labels,
                                             tokenizer)

        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)
        self.metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels,
        )

    def compute_acc(self):
        result = self.metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result

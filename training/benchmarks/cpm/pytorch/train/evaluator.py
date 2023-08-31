import torch

import numpy as np
from train.metrics import average_corpus_level
from model.losses.cross_entropy import cross_entropy
from model.fp16 import FP16_Module
from driver import dist_pytorch


class Evaluator:

    def __init__(self, config, dataloader):
        self.config = config
        self.eval_dataloader = dataloader

    def evaluate(self, trainer):
        model = trainer.model
        device = trainer.device
        world_size = dist_pytorch.get_world_size()

        model.eval()
        all_losses = []
        all_embedding_average = []
        with torch.no_grad():
            for _, data in enumerate(self.eval_dataloader):
                batch, no_model_batch = data[0], data[1]
                for k in batch:
                    batch[k] = batch[k].to(device)
                for k in no_model_batch:
                    no_model_batch[k] = no_model_batch[k].to(device)

                output = model(**batch)
                labels = no_model_batch["labels"]

                # losses [b,s]
                losses = cross_entropy(output.contiguous().float(), labels)
                loss_mask = no_model_batch["loss_mask"].view(-1)
                #loss 标量
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
                all_losses.append(loss.item())

                preds = torch.argmax(output, -1)
                if not hasattr(model, "module"):
                   embeddings = model.word_embeddings.weight
                elif isinstance(model.module, FP16_Module):
                    embeddings = model.module.module.word_embeddings.weight
                else:
                    embeddings = model.module.word_embeddings.weight

                #embedding_average 形状是[batch_size]
                embedding_average = average_corpus_level(
                    preds.cpu(), labels.cpu(), embeddings.cpu(),
                    no_model_batch["loss_mask"].cpu())
                all_embedding_average.append(embedding_average.mean)

        model.train()

        all_embedding_average_tensor = torch.tensor(
            np.mean(all_embedding_average), dtype=torch.float32, device=device)
        all_losses_tensor = torch.tensor(np.mean(all_losses),
                                         dtype=torch.float32,
                                         device=device)

        if torch.distributed.is_initialized():
            # Collect total scores from all ranks
            torch.distributed.all_reduce(all_losses_tensor,
                                         op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(all_embedding_average_tensor,
                                         op=torch.distributed.ReduceOp.SUM)

        # Average by number of examples
        all_losses_tensor /= world_size
        all_embedding_average_tensor /= world_size

        return all_losses_tensor.item(), all_embedding_average_tensor.item()

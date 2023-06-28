import torch as th

from utils.tensor import reduce_tensor
from driver import dist_pytorch


class Evaluator:

    def __init__(self, config):
        super(Evaluator, self).__init__()
        self.config = config

    # predict the tail for (head, rel, -1) or head for (-1, rel, tail)
    def predict(self,
                model,
                graph,
                device,
                data_iter,
                split="valid",
                mode="tail"):
        model.eval()
        config = self.config
        with th.no_grad():
            results = {}
            train_iter = iter(data_iter["{}_{}".format(split, mode)])

            for step, batch in enumerate(train_iter):
                triple, label = batch[0].to(device), batch[1].to(device)
                sub, rel, obj, label = (
                    triple[:, 0],
                    triple[:, 1],
                    triple[:, 2],
                    label,
                )
                pred = model(graph, sub, rel)
                b_range = th.arange(pred.size()[0], device=device)
                target_pred = pred[b_range, obj]
                pred = th.where(label.byte(), -th.ones_like(pred) * 10000000,
                                pred)
                pred[b_range, obj] = target_pred

                # compute metrics
                ranks = (1 + th.argsort(
                    th.argsort(pred, dim=1, descending=True),
                    dim=1,
                    descending=False,
                )[b_range, obj])
                ranks = ranks.float()

                reduced_ranks = ranks

                if dist_pytorch.is_dist_avail_and_initialized():
                    th.distributed.barrier()
                    reduced_ranks = reduce_tensor(ranks, config.n_device)

                size = ranks.shape[0]

                results["count"] = th.numel(reduced_ranks) + results.get(
                    "count", 0.0)
                results["mr"] = th.sum(reduced_ranks).item() + results.get(
                    "mr", 0.0)
                results["mrr"] = th.sum(
                    1.0 / reduced_ranks).item() + results.get("mrr", 0.0)
               
                for k in [1, 3, 10]:
                    results["hits@{}".format(k)] = th.numel(
                        reduced_ranks[reduced_ranks <= (k)]) + results.get(
                            "hits@{}".format(k), 0.0)

        return results

    # evaluation function, evaluate the head and tail prediction and then combine the results
    def evaluate(self, model, graph, device, data_iter, split="valid"):
        # predict for head and tail
        left_results = self.predict(model,
                                    graph,
                                    device,
                                    data_iter,
                                    split,
                                    mode="tail")
        right_results = self.predict(model,
                                     graph,
                                     device,
                                     data_iter,
                                     split,
                                     mode="head")
        results = {}
        count = float(left_results["count"])

        # combine the head and tail prediction results
        # Metrics: MRR, MR, and Hit@k
        results["left_mr"] = round(left_results["mr"] / count, 5)
        results["left_mrr"] = round(left_results["mrr"] / count, 5)
        results["right_mr"] = round(right_results["mr"] / count, 5)
        results["right_mrr"] = round(right_results["mrr"] / count, 5)
        results["mr"] = round(
            (left_results["mr"] + right_results["mr"]) / (2 * count), 5)
        results["mrr"] = round(
            (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5)

        for k in [1, 3, 10]:
            results["left_hits@{}".format(k)] = round(
                left_results["hits@{}".format(k)] / count, 5)
            results["right_hits@{}".format(k)] = round(
                right_results["hits@{}".format(k)] / count, 5)
            results["hits@{}".format(k)] = round(
                (left_results["hits@{}".format(k)] +
                 right_results["hits@{}".format(k)]) / (2 * count),
                5,
            )
        return results

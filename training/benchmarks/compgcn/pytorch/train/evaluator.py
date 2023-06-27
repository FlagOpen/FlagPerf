import torch as th

from utils.tensor import reduce_tensor
from utils.meter import AverageMeter
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
            # count = AverageMeter('count', ':10d')
            mrr = AverageMeter('mrr', ':8.5f')
            mr = AverageMeter('mr', ':8.5f')
            hits1 = AverageMeter('hits1', ':8.5f')
            hits3 = AverageMeter('hits3', ':8.5f')
            hits10 = AverageMeter('hits10', ':8.5f')

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

                # print(f"step:{step} size:{size} ranks:{ranks} reduced_ranks:{reduced_ranks}")

                results["count"] = th.numel(reduced_ranks) + results.get(
                    "count", 0.0)
                # count.update(th.numel(reduced_ranks))

                results["mr"] = th.sum(reduced_ranks).item() + results.get(
                    "mr", 0.0)
                # mr.update(th.sum(reduced_ranks).item(), size)
                results["mrr"] = th.sum(
                    1.0 / reduced_ranks).item() + results.get("mrr", 0.0)
                # mrr.update(th.sum(1.0 / reduced_ranks).item(), size)

                # print(f"config.n_device:{config.n_device} tmp_count:{th.numel(reduced_ranks)} size:{size} count:{mr.count} \
                #       tmp_mrr:{th.sum(1.0 / reduced_ranks).item()} temp_mr:{th.sum(reduced_ranks).item()}" )

                # hits1.update(th.numel(reduced_ranks[reduced_ranks <= (1)]), size)
                # hits3.update(th.numel(reduced_ranks[reduced_ranks <= (3)]), size)
                # hits10.update(th.numel(reduced_ranks[reduced_ranks <= (10)]), size)

                # results['count'] = mrr.count
                # results['mr'] = mr.avg
                # results['mrr'] = mrr.avg
                # results['hits@1'] = hits1.avg
                # results['hits@3'] = hits3.avg
                # results['hits@10'] = hits10.avg
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

        # print(f"evaluate left_results count: {count}")

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

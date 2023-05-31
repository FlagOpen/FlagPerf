import numpy as np
import torch


class Evaluator:

    def __init__(self, config, eval_dataloader):
        super(Evaluator, self).__init__()
        self.config = config
        self.data_loader = eval_dataloader

    def evaluate(self, model, device):
        model.eval()

        print("Running validation...")

        all_c, all_top1, all_top5 = [], [], []
        step = 0
        for b, (x, y) in enumerate(self.data_loader):
            with torch.no_grad():
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1, top5 = self.topk(logits, y, ks=(1, 5))
                all_c.extend(c.cpu())  # Also ensures a sync point.
                all_top1.extend(top1.cpu())
                all_top5.extend(top5.cpu())
                if step % self.config.print_freq == 0:
                    print(step, end='/')
                    print(len(self.data_loader))
                step += 1

        model.train()
        print(f"Validation loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")

        return all_c, all_top1, all_top5

    def topk(self, output, target, ks=(1, )):
        """Returns one boolean vector for each k, whether the target is within the output's top-k."""
        _, pred = output.topk(max(ks), 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].max(0)[0] for k in ks]

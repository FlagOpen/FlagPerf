class Evaluator:

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.args = args

    def evaluate(self, trainer):
        itr = self.dataloader.next_epoch_itr(shuffle=False)
        subset_losses = []
        for sample in itr:
            loss = trainer.valid_step(sample)
            subset_losses.append(loss)
        loss = sum(subset_losses) / (len(subset_losses) or 1)

        return loss

import time
import torch
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter
from utils.utils import reduce_tensor

class Evaluator:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    @torch.no_grad()
    def evaluate(self, config, model):
        dataloader = self.dataloader
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        end = time.time()
        for idx, (images, target) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=config.amp_enable):
                output = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if idx % config.print_freq == 0:
                print("-----------eval_acc1:",acc1_meter.avg)
            #     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                # logger.info(
                #     f'Test: [{idx}/{len(data_loader)}]\t'
                #     f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #     f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                #     f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                #     f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                #     f'Mem {memory_used:.0f}MB')
        # logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

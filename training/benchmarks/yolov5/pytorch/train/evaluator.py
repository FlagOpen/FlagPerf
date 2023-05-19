import torch
import torch.distributed as dist
import val

class Evaluator:

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.args = args
        self.total_loss = 0.0
        
        # results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.results = (0, 0, 0, 0, 0, 0, 0)
        
        self.P = 0
        self.R = 0
        self.mAP_5 = 0
        self.mAP_5_95 = 0
        self.eval_lbox = 0
        self.eval_lobj = 0
        self.eval_lcls = 0
        # map
        self.total_batch = 0

    def __update(self, loss, P, R,mAP_5,mAP_5_95,eval_lbox,eval_lobj,eval_lcls):
        self.total_loss += loss
        self.P += P 
        self.R += R
        self.mAP_5 += mAP_5 
        self.mAP_5_95 += mAP_5_95
        self.eval_lbox += eval_lbox 
        self.eval_lobj += eval_lobj 
        self.eval_lcls += eval_lcls
        
        # self.total_acc1 += acc1
        # self.total_acc5 += acc5
        
        self.total_batch += 1

    def evaluate(self, trainer):
        # yolov5 evaluate methods
        # map and loss... 
        self.total_loss, self.total_eval_lbox, self.total_eval_lobj, self.total_eval_lcls = 0.0, 0.0, 0.0, 0.0
        self.total_batch = 0
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                batch = trainer.process_batch(batch, self.args.device)
                loss, eval_lbox, eval_lobj, eval_lcls  = trainer.inference(batch)
                self.__update(loss.item(), eval_lbox.item(), eval_lobj.item(), eval_lcls.item())

        if dist.is_available() and dist.is_initialized():
            total = torch.tensor([
                self.total_loss, self.total_eval_lbox, self.total_eval_lobj, self.total_eval_lcls, 
                self.total_batch
            ],
                                 dtype=torch.float32,
                                 device=self.args.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            self.total_loss, self.total_eval_lbox, self.total_eval_lobj, self.total_eval_lcls, self.total_batch = total.tolist(
            )

        loss = self.total_loss / self.total_batch
        eval_lbox = self.total_eval_lbox / self.total_batch
        eval_lobj = self.total_eval_lobj / self.total_batch
        eval_lcls = self.total_eval_lcls / self.total_batch 
        
        return loss, eval_lbox, eval_lobj, eval_lcls 
    
    # def evaluate_(self):
    #         # EMA
    #     ema = ModelEMA(model) if RANK in {-1, 0} else None
    #     results, maps, _ = val.run(data_dict,
    #                         batch_size=batch_size // WORLD_SIZE * 2,
    #                         imgsz=imgsz,
    #                         half=amp,
    #                         model=ema.ema,
    #                         single_cls=single_cls,
    #                         dataloader=val_loader,
    #                         save_dir=save_dir,
    #                         plots=False,
    #                         callbacks=callbacks,
    #                         compute_loss=compute_loss)
    #     return 0

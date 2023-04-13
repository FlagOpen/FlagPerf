from model import create_model
import time
import os
import torch
from collections import OrderedDict
import torch.nn as nn

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm import utils
from timm.models import safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy


class Trainer:

    def __init__(self, device, args):
        super(Trainer, self).__init__()
        self.device = device
        self.args = args
        # self.driver = driver
        # self.adapter = adapter
        # self.training_state = training_state
        # self.grad_scaler = None

        # self.device = device
        self.optimizer = None
        # self.config = config
        self.model = None
        # self.evaluator = evaluator
        self.lr_scheduler = None
        # self.global_batch_size = None
        # self.overflow_buf = None
        self.train_loss_fn = None
        self.validate_loss_fn = None
        num_aug_splits = 0
        if args.aug_splits > 0:
            assert args.aug_splits > 1, 'A split of 1 makes no sense'
            num_aug_splits = args.aug_splits
        self.num_aug_splits = num_aug_splits
        self.mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        
    def init(self):
        self.model = create_model("vit_base_patch16_224")
        self.model = self._init_model(self.model, self.args, self.device)
        
        # self.model = self.adapter.convert_model(self.model)
        # self.model = self.adapter.model_to_fp16(self.myesodel)
        self.optimizer = self._create_optimizer(
            self.model,
            self.args,
        )
        self.train_loss_fn = self._init_train_loss_fn(self.args)
        self.validate_loss_fn = self._init_validate_loss_fn(self.args)
        # self.model = self.adapter.model_to_ddp(self.model)
    #     self.lr_scheduler = create_scheduler(
    #     self.optimizer,
    #     updates_per_epoch=updates_per_epoch,
    #     args=self.args,
    # )
        # self.grad_scaler = self.adapter.create_grad_scaler()
        
    def _init_model(self, model, args, device):
        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

        if args.grad_checkpointing:
            model.set_grad_checkpointing(enable=True)

        if utils.is_primary(args):
            print(
                f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')


        # setup augmentation batch splits for contrastive loss or split bn


        # enable split bn (separate bn stats per batch-portion)
        if args.split_bn:
            assert self.num_aug_splits > 1 or args.resplit
            model = convert_splitbn_model(model, max(self.um_aug_splits, 2))

        # move model to GPU, enable channels last layout if set
        model.to(device=device)
        if args.channels_last:
            model.to(memory_format=torch.channels_last)

        # setup synchronized BatchNorm for distributed training
        if args.distributed and args.sync_bn:
            args.dist_bn = ''  # disable dist_bn when sync BN active
            assert not args.split_bn
            if has_apex and use_amp == 'apex':
                # Apex SyncBN used with Apex AMP
                # WARNING this won't currently work with models using BatchNormAct2d
                model = convert_syncbn_model(model)
            else:
                model = convert_sync_batchnorm(model)
            if utils.is_primary(args):
                print(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        if args.torchscript:
            assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            model = torch.jit.script(model)
        elif args.torchcompile:
            # FIXME dynamo might need move below DDP wrapping? TBD
            assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
            torch._dynamo.reset()
            model = torch.compile(model, backend=args.torchcompile)
        elif args.aot_autograd:
            assert has_functorch, "functorch is needed for --aot-autograd"
            model = memory_efficient_fusion(model)
               
        return model
    
    def _create_optimizer(self, model, args):
        if not args.lr:
            global_batch_size = args.batch_size * args.world_size
            batch_ratio = global_batch_size / args.lr_base_size
            if not args.lr_base_scale:
                on = args.opt.lower()
                args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
            if args.lr_base_scale == 'sqrt':
                batch_ratio = batch_ratio ** 0.5
            args.lr = args.lr_base * batch_ratio
            if utils.is_primary(args):
                print(
                    f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                    f'and global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')
        optimizer = create_optimizer_v2(
            model,
            **optimizer_kwargs(cfg=args),
            **args.opt_kwargs,
        )
        return optimizer
    
    def _init_train_loss_fn(self, args):
        # setup loss function
        if args.jsd_loss:
            assert self.num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=self.num_aug_splits, smoothing=args.smoothing)
        elif self.mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing:
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=self.device)
        return train_loss_fn
    
    def _init_validate_loss_fn(self, args):
        validate_loss_fn = nn.CrossEntropyLoss().to(device=self.device)
        return validate_loss_fn
    
    def train_one_epoch(
        self,
        epoch,
        loader,
        # loss_fn,
        # args,
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=None,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None
    ):
        device = self.device
        args=self.args
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if args.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        batch_time_m = utils.AverageMeter()
        data_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()

        self.model.train()

        end = time.time()
        num_batches_per_epoch = len(loader)
        last_idx = num_batches_per_epoch - 1
        num_updates = epoch * num_batches_per_epoch

        loss_list = []
        for batch_idx, (input, target) in enumerate(loader):

            if batch_idx >= 100:
                break

            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            if not args.prefetcher:
                input, target = input.to(device), target.to(device)
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = self.model(input)
                loss = self.train_loss_fn(output, target)

            self.optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, self.optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(self.model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order
                )
            else:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    utils.dispatch_clip_grad(
                        model_parameters(self.model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad,
                        mode=args.clip_mode
                    )
                self.optimizer.step()

            if model_ema is not None:
                model_ema.update(self.model)

            # save loss and map for check
            if epoch == 0 and batch_idx == 0 and device.type == "xla":
                dname = "p_grad"
                if not os.path.isdir(dname):
                    os.makedirs(dname)
                dname = dname + "/" + device.type
                if os.path.isdir(dname):
                    import shutil
                    shutil.rmtree(dname)
                os.makedirs(dname)
                for pi, p in enumerate(self.model.parameters()):
                    if p.grad is not None:
                        file_name = dname + "/" + "p_grad_" + str(pi)
                        torch.save(p.grad.cpu(), file_name)
            
            # 
            import torch_xmlir.core.xpu_model as xm
            # xm.mark_step(None)  # eager模式不需要

            if not args.distributed:
                losses_m.update(loss.item(), input.size(0))
            
            loss_list.append(loss.item())  # scalar tensor

            # torch.cuda.synchronize()

            num_updates += 1
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                    losses_m.update(reduced_loss.item(), input.size(0))

                if utils.is_primary(args):
                    print(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m)
                    )

                    if args.save_images and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                            padding=0,
                            normalize=True
                        )

            if saver is not None and args.recovery_interval and (
                    last_batch or (batch_idx + 1) % args.recovery_interval == 0):
                saver.save_recovery(epoch, batch_idx=batch_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
            # end for

        # save loss and map for check
        dname = "check"
        if not os.path.isdir(dname):
            os.makedirs(dname)
        dname = dname +"/" + device.type
        if not os.path.isdir(dname):
            os.makedirs(dname)
        torch.save(torch.tensor(loss_list), dname + "/loss_" + str(0) + ".pt")

        import torch_xmlir.debug.metrics as met
        print(met.metrics_report())

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])
    
    def train_one_step(self, batch):
        # todo
        return
        

        
    
    def forward(self, batch):
        
        return 
    
    def inference(self, batch):
        return 

    def process_batch(self, batch, device):
        return 
            

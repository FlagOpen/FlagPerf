import logging
import torch
import math
import inspect
from torch import nn
from multiprocessing import Semaphore
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
import torch.distributed as dist
import transformers
from transformers import (
    TrainingArguments,
)
from collections.abc import Mapping
from transformers.training_args import OptimizerNames
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.optimization import Adafactor, get_scheduler
from transformers.trainer_pt_utils import get_model_param_count,IterableDatasetShard,LabelSmoother,get_parameter_names
from transformers.trainer_utils import seed_worker,has_length,get_last_checkpoint,RemoveColumnsCollator
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import (
    is_datasets_available,
    is_torch_xla_available,
    find_labels,
)
from mstt_util import(
    PrecisionDebuggerINIT,
    PrecisionDebuggerMarkStep,
    PrecisionDebuggerBGN,
    PrecisionDebuggerEND,
)
from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger(__name__)
logger.setLevel(20)

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla.distributed.fsdp import (
        consolidate_sharded_model_checkpoints,
        checkpoint_module,
    )
import os

class XlaTrainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        model_args = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        tsprobe_config: Optional[str] = None,
        dump_path: Optional[str] = None,   
        gloal_semaphore: Optional[Semaphore] = None,        
    ):
        self.args = args
        self.model_args = model_args
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.model = model
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.lr_scheduler = optimizers
        self._train_batch_size = args.train_batch_size
        self._signature_columns = None
        self.gloal_semaphore = gloal_semaphore
        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
        self.debugger = None
        if tsprobe_config:
            from tsprobe.pytorch import PrecisionDebugger
            self.debugger = PrecisionDebugger(config_path=tsprobe_config,
                                              level = 'L1',
                                              model = model,
                                              dump_path = dump_path,
                                              step=None)
        if self.model_args.mstt_config_name_or_path:
            PrecisionDebuggerINIT(self.model_args.mstt_config_name_or_path,model=self.model)

    def _get_learning_rate(self):
        last_lr = self.optimizer.param_groups[0]["lr"]
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
        return last_lr

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["labels", "input_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )
        return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        total_rank = self.model_args.ddp_dp_sharding * self.model_args.megatron_tp_sharding
        total_tensor = torch.tensor(range(total_rank))

        if self.model_args.use_cuda:
            all_reduce_sharding_groups = []
            for i in range(self.model_args.ddp_dp_sharding):
                all_reduce = total_tensor[i*self.model_args.megatron_tp_sharding:(i+1)*self.model_args.megatron_tp_sharding]
                all_reduce_sharding_groups.extend(all_reduce.view(1,-1).tolist())
        else:
            dp_tensor = total_tensor.view(-1, self.model_args.ddp_dp_sharding)
            all_reduce_sharding_groups = []
            for i in range(dp_tensor.size()[0]):
                dp_i = dp_tensor[i]
                all_gather = dp_i.view(self.model_args.megatron_tp_sharding, -1)
                all_reduce = all_gather.transpose(0, 1)
                all_reduce_sharding_groups.extend(all_reduce.tolist())
        def get_rank():
            id = dist.get_rank()
            print('all_reduce_sharding_groups:', all_reduce_sharding_groups)
            for i, tp_group in enumerate(all_reduce_sharding_groups):
                if id in tp_group:
                    return i
            else:
                raise ValueError(f"Not Support all_reduce_sharding_groups:{all_reduce_sharding_groups}.")
        num_replicas=1 if not dist.is_initialized() else dist.get_world_size() // self.model_args.megatron_tp_sharding
        rank=0 if not dist.is_initialized() else get_rank()
        return DistributedSampler(self.train_dataset, num_replicas=num_replicas, rank=rank, shuffle=False)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        # TODO(jonbolin): Disabling Accelerate on the dataloader. Accelerate does not
        # currently support SPMD-mode
        return DataLoader(train_dataset, **dataloader_params)

    def _xla_sharded_dataloader(self, dataloader):
        if is_torch_xla_available():
            import torch_xla.experimental.xla_sharding as xs
            import torch_xla.distributed.parallel_loader as pl
            sharding_spec = xs.ShardingSpec(self.args.spmd_mesh, (0, None))
            # TODO(jonbolin): Once integrated with Accelerate, we can use the Accelerate-prepared
            # MpDeviceLoader instead of manually adding sharding and adding a dataset attribute.
            loader = pl.MpDeviceLoader(dataloader, self.args.device, input_sharding=sharding_spec, loader_prefetch_size=self.args.train_batch_size, device_prefetch_size=4)
            loader.dataset = dataloader.dataset
            return loader
        else:
            return dataloader

    def num_examples(self, dataloader: DataLoader) -> int:
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            if self.model_args.use_cuda:
                return data.to('cuda')
            else:
                kwargs = {"device": self.args.device}
                return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        return inputs

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments, model_args=None) -> Tuple[Any, Any]:
        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim in [OptimizerNames.ADAMW_TORCH, 'adamw_torch_tx8_fused']:
            if args.optim == OptimizerNames.ADAMW_TORCH:
                if not model_args.use_cuda:
                    from torch_adamw import AdamW
                else:
                    from torch.optim import AdamW
            else:
                from tx8_adamw import AdamW
            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def get_decay_parameter_names(self, model) -> List[str]:
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = XlaTrainer.get_optimizer_cls_and_kwargs(self.args, self.model_args)
        if self.args.optim == 'adamw_torch_tx8_fused':
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, amp_fp32_param_groups= self.amp_fp32_param_groups, **optimizer_kwargs)
        else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        self.debugger.start() if self.debugger else None
        outputs = model(**inputs)
        self.debugger.forward_backward_dump_end() if self.debugger else None
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        # xm.mark_step()
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        # xm.mark_step()
        model.set_grad_all_reduce() if hasattr(model, 'set_grad_all_reduce') else None
        return loss.detach() # / self.args.gradient_accumulation_steps

    def train(
        self,
        **kwargs,
    ):
        args = self.args
        self._train_batch_size = self.args.train_batch_size

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Initialize optimizer
        self.gloal_semaphore.acquire() if self.gloal_semaphore is not None else None
        try:    
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)      
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.grad = torch.autograd.Variable(param.data.new(param.size()).zero_())
                    param.grad.requires_grad_(False)
            self.optimizer.step()
            self.lr_scheduler.step()
            # get cpu optimizer param use to send tx8
            # self.get_optim_amp_param(self.optimizer)  
            self.model.zero_grad()
            state_dict = self.optimizer.state_dict()
            if self.model_args.use_cuda:
                self.model = self.model.to('cuda')
            else:
                self.model = self.model.to(self.args.device)
            self.optimizer = None
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            self.optimizer.load_state_dict(state_dict)
            logger.info('!!!!!!!!!!!!!!!optim initialization done')
        finally:
            self.gloal_semaphore.release() if self.gloal_semaphore is not None else None
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")
        
        start_epoch = 0
        start_step = -1
        
        if args.resume_from_checkpoint is not None:
            self.checkpoint_path = f'{self.checkpoint_path}/tmp_rank-{xm.get_ordinal()}-of-{xm.xrt_world_size()}.pth'
            # 尝试加载检查点
            if os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path)
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']   # 从上一个 epoch 开始继续训练
                start_step = checkpoint['current_step']
                
                if start_step == -1:
                    start_epoch += 1
                print(f"恢复训练，继续从第 {start_epoch} 个 epoch {start_step} 个 step 开始。")

        self.model.train()
        # 初始化tensorboard
        global_step = 0
        try:
            rank = dist.get_rank()
        except:
            rank=0
        writer = SummaryWriter(log_dir=f"{args.output_dir}/rank-{rank}")
        tr_loss = 0
        for epoch in range(start_epoch, num_train_epochs):
            epoch_iterator = train_dataloader
            self.optimizer.zero_grad()
            for step, inputs in enumerate(epoch_iterator):
                if step <= start_step:
                    continue
                #inputs = torch.load("input.pt")
                #PrecisionDebuggerBGN()
                tr_loss_step = self.training_step(self.model, inputs)
                tr_loss += tr_loss_step
                global_step += 1
                #PrecisionDebuggerEND()
                if global_step % args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if not self.model_args.use_cuda:
                    xm.mark_step()
                    PrecisionDebuggerMarkStep() if self.model_args.mstt_config_name_or_path else None

                if global_step % args.gradient_accumulation_steps == 0:
                    self.lr_scheduler.step()
                    learning_rate = self._get_learning_rate()
                    print(f"!!!!!!!!! rank {rank}, do step : {global_step-1}, tr_loss:{tr_loss},learning_rate:{learning_rate}")
                    # 写入tensorboard
                    # writer.add_scalar('train/learning_rate', learning_rate, global_step-1)
                    # writer.add_scalar('train/loss', tr_loss, global_step-1)
                    tr_loss = 0

                self.debugger.stop() if self.debugger else None
                self.debugger.step() if self.debugger else None                    

                if args.resume_from_checkpoint is not None and step % args.save_steps == 0:
                    print("save ckpt current_step: ", step, " args.save_steps: ", args.save_steps)
                    ckpt = {
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                        'current_step': step,
                        'shard_metadata': self.model.get_shard_metadata(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    xm.save(ckpt, self.checkpoint_path, master_only=False)
                    
                    # Consolidate the sharded model checkpoints 
                    if xm.is_master_ordinal(local=False):
                        consolidate_sharded_model_checkpoints(
                            ckpt_prefix=f'{self.resume_from_checkpoint}', ckpt_suffix="_rank-*-of-*.pth")
                    xm.rendezvous('ckpt_consolidation')
                            
            if args.resume_from_checkpoint is not None:
                # 每一轮训练完成后保存权重
                ckpt = {
                            'model': self.model.state_dict(),
                            'epoch': epoch,
                            'current_step': -1,
                            'shard_metadata': self.model.get_shard_metadata(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                xm.save(ckpt, self.checkpoint_path, master_only=False)
                
                # Consolidate the sharded model checkpoints 
                if xm.is_master_ordinal(local=False):
                    consolidate_sharded_model_checkpoints(
                        ckpt_prefix=f'{self.resume_from_checkpoint}', ckpt_suffix="_rank-*-of-*.pth")
                xm.rendezvous('ckpt_consolidation')
            

import logging
import torch
import math
import inspect
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import transformers
from transformers import (
    TrainingArguments,
)
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_model_param_count,IterableDatasetShard,LabelSmoother
from transformers.trainer_utils import seed_worker,has_length,get_last_checkpoint,RemoveColumnsCollator
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import (
    is_datasets_available,
    is_torch_xla_available,
    find_labels,
)

from mstt_util import PrecisionDebuggerBGN,PrecisionDebuggerEND,PrecisionDebuggerINIT,PrecisionDebuggerMarkStep
logger = logging.getLogger(__name__)
logger.setLevel(20)

if is_datasets_available():
    import datasets

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

class XlaTrainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        self.args = args
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
        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
        PrecisionDebuggerINIT('./config_tensor.json',model=self.model)

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
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

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
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return RandomSampler(self.train_dataset)

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

    def create_optimizer_and_scheduler(self):
        import torch.optim as optim
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!do forward training done")
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        return loss.detach() / self.args.gradient_accumulation_steps

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
        self.create_optimizer_and_scheduler()

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
        self.model = self.model.to(self.args.device)
        self.model.train()
        global_step = 0
        learning_rate = 0
        try:
            rank = torch.distributed.get_rank()
        except:
            rank=0
        for epoch in range(0, 1):
            epoch_iterator = train_dataloader
            for step, inputs in enumerate(epoch_iterator):
                self.optimizer.zero_grad()
                tr_loss_step = self.training_step(self.model, inputs)
                self.optimizer.step()
                print(f"!!!!!!!!!!!!!!!!do mark_step")
                xm.mark_step()
                global_step += 1
                #learning_rate = self._get_learning_rate()
                print(f"!!!!!!!!! rank {rank}, do step : {step}, tr_loss_step:{tr_loss_step},learning_rate:{learning_rate}")
                if step==0:
                    break

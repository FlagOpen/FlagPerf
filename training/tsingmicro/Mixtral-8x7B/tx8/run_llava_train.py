import os
import logging
import torch
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import transformers
import numpy as np
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    is_torch_available,
    is_vision_available,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    # TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import torch_xla.experimental.pjrt_backend  
import torch_xla.experimental.pjrt as pjrt
import torch_xla.experimental.xla_sharding as xs

check_min_version("4.41.0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

os.environ['TX8_MODEL_EXPORT_LEVEL']="10"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

from torch.utils.data import Dataset
from typing import Dict, List, Optional
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        pixel_values = torch.randn(
            (2, 3, 336, 336),
            dtype=torch.float,
        )
        input_ids = torch.tensor(
            [
                [32001, 32001, 1, 15043, 7084, 32000, 29871, 13, 7900],
                [1, 15043, 7084, 29901, 29871, 32000, 29871, 13, 7900],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.long,
        )
        self.pixel_values = pixel_values
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return 4

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            pixel_values=self.pixel_values,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    
    is_fsdp_xla_enabled = training_args.fsdp_config["xla"]
    num_devices = pjrt.global_device_count()
    if is_fsdp_xla_enabled:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp import checkpoint_module
        from torch_xla.distributed.fsdp.wrap import size_based_auto_wrap_policy,transformer_auto_wrap_policy
        import functools
        from transformers.trainer_pt_utils import get_module_class_from_name
 
        auto_wrap_policy = None
        auto_wrapper_callable = None
        default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = training_args.fsdp_config.get("transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap)
        
        if fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception("Could not find the transformer layer class to wrap in the model.")
                else:
                    transformer_cls_to_wrap.add(transformer_cls)

            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_cls_to_wrap,
            )
        fsdp_kwargs = training_args.xla_fsdp_config
        if training_args.fsdp_config["xla_fsdp_grad_ckpt"]:
            # Apply gradient checkpointing to auto-wrapped sub-modules if specified
            def auto_wrapper_callable(m, *args, **kwargs):
                return FSDP(checkpoint_module(m), *args, **kwargs)
        
        fsdp_wrap = lambda m: FSDP(m,auto_wrap_policy=auto_wrap_policy,auto_wrapper_callable=auto_wrapper_callable,flatten_parameters=True,**fsdp_kwargs,)
        #fsdp_wrap = lambda m: FSDP(m,compute_dtype=torch.bfloat16, shard_param_on_dim_0=True, pin_layout_in_collective_ops=True, disable_reshard_on_root=False)
        import inspect
        forward_signature = inspect.signature(model.forward.__func__)
        model = fsdp_wrap(model)
        model.forward.__func__.__signature__ = forward_signature
        print(f"!!!!modle: {model}")
    elif num_devices>1:
        print(f"!!!!!!!!num_devices:{num_devices}")
        mesh_shape = (2, 2)
        spmd_mesh = xs.Mesh(np.arange(num_devices), mesh_shape)
        model = model.to(xm.xla_device())
        for name, param in model.named_parameters(): 
            if len(param.shape) == 1:
                continue      
            # Apply 2D sharding:
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                xs.mark_sharding(param, spmd_mesh, (1, None))
            elif 'o_proj' in name:
                xs.mark_sharding(param, spmd_mesh, (None, 1))
            elif 'gate_proj' in name or 'up_proj' in name:
                xs.mark_sharding(param, spmd_mesh, (1, None))
            elif 'down_proj' in name:
                xs.mark_sharding(param, spmd_mesh, (None, 1))
            print(f'!!!!!!!!{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}')
    else:
        print(f"!!!!!!one host one cards train")

    # Simulate some user inputs
    torch_device = xm.xla_device()
    pixel_values = torch.randn(
        (1, 3, 336, 336),
        dtype=torch.float,
        device=torch_device,
    )
    input_ids = torch.tensor(
        [
            [1, 15043, 7084, 29901, 29871, 32000, 29871, 13, 7900],
        ],
        dtype=torch.long,
        device=torch_device,
    )
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.long,
        device=torch_device,
    )

    model = model.to(torch_device)
    model.train()
    
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
    optimizer = optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    train_dataset = SupervisedDataset()
    def get_train_dataloader(train_dataset) -> DataLoader:
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        dataloader_params = {
            "batch_size": 1,
        }

        return DataLoader(train_dataset, **dataloader_params)
    def _xla_sharded_dataloader(dataloader):
        if is_torch_xla_available():
            import torch_xla.experimental.xla_sharding as xs
            import torch_xla.distributed.parallel_loader as pl
            sharding_spec = xs.ShardingSpec(spmd_mesh, (0, None)) 
            # TODO(jonbolin): Once integrated with Accelerate, we can use the Accelerate-prepared
            # MpDeviceLoader instead of manually adding sharding and adding a dataset attribute.
            loader = pl.MpDeviceLoader(dataloader, training_args.device, input_sharding=sharding_spec, loader_prefetch_size=training_args.train_batch_size, device_prefetch_size=4)
            loader.dataset = dataloader.dataset
            return loader
        else:
            return dataloader
    #train_dataloader = _xla_sharded_dataloader(get_train_dataloader(train_dataset))
    train_dataloader = get_train_dataloader(train_dataset)
    # Make sure that the loss is properly computed
    for step, inputs in enumerate(train_dataloader):
        #logger.info('input sharding', {k: (v.shape, torch_xla._XLAC._get_xla_sharding_spec(v)) for k, v in inputs.items()})
        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss.backward()
        optimizer.step()
        xm.mark_step()
        print(f"!!!!!!!!!!!loss: %s" % loss.detach())
        raise


def _mp_fn(index):
    # For xla_spawn (TPUs)
    print(f"!!!!!!rank:{index},CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    dist.init_process_group('xla', init_method='pjrt://')
    main()

if __name__ == "__main__":
    main()

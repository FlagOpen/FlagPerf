import os
import sys
import logging
import torch
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
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
    TrainingArguments,
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
from tx8 import tx8_replace_func
from transformers.optimization import Adafactor, get_scheduler
from transformers.training_args import OptimizerNames
from transformers.trainer_pt_utils import get_model_param_count,IterableDatasetShard,LabelSmoother,get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
#mstt中间数据
# from mstt_util import PrecisionDebuggerBGN,PrecisionDebuggerEND,PrecisionDebuggerINIT,PrecisionDebuggerMarkStep
# PrecisionDebuggerINIT(config_path="/login_home/zhoujunjie/zhoujunjie_transformers/config_tensor.json")


check_min_version("4.41.0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import random
DEFAULT_RANDOM_SEED = 42

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# basic + torch
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
    # XLA
    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)

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
    mstt_config_name_or_path: str = field(
        default=None,
    )
    llama_num_hidden_layers: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model block nums."
            )
        },
    )
    clip_num_hidden_layers: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model block nums."
            )
        },
    )
    use_cuda: bool = field(
        default=False, metadata={"help": "Whether to use cuda or xla."}
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

    json_file_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    images_path: Optional[str] = field(
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

        if self.json_file_path is None and self.train_file is None and self.validation_file is None:
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
import json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, json_path, image_root, processor,max_text_length, use_cuda = False):
        super(SupervisedDataset, self).__init__()

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.processor = processor
        self.max_text_length = max_text_length
        self.use_cuda = use_cuda

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        user_prompt = next(
            (conv["value"] for conv in item["conversations"] if conv["from"] == "human"),
            None
        )
        #从读取的内容中删除<image>，避免生成的input_ids中有多个32000
        user_prompt = user_prompt.replace('<image>' , '').replace("\n","")

        if user_prompt is None:
            raise ValueError(f"Data ID {item['id']} - No 'human' prompt found.")

        # 提取 gpt 标签 (label)
        gpt_response = next(
            (conv["value"] for conv in item["conversations"] if conv["from"] == "gpt"),
            None
        )

        if gpt_response is None:
            raise ValueError(f"Data ID {item['id']} - No 'gpt' response found.")

        # 拼接完整的图像路径
        image_file = os.path.join(self.image_root, item["image"])
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")

        # 加载图像
        raw_image = Image.open(image_file).convert("RGB")

        # 拼接 prompt
        prompt = f"{user_prompt}\n<image>"

        if self.use_cuda:
            torch_device = 'cuda'
        else:
            torch_device = xm.xla_device() 
        # 处理输入（图像 + 文本）
        inputs = self.processor(
            images=raw_image,
            text=prompt,
            return_tensors="pt",
            max_length=self.max_text_length,
            padding="max_length",
            padding_side="right",
            truncation=True
        ).to(torch_device)

        # 转换标签为 token_ids
        labels = self.processor(
            text=gpt_response,
            return_tensors="pt",
            max_length=self.max_text_length,
            padding="max_length",
            padding_side="right",
            truncation=True
        ).input_ids.to(torch_device)
        inputs["labels"] = labels 
        
        return inputs

#获取优化器参数
@staticmethod
def get_optimizer_cls_and_kwargs(args: TrainingArguments, model_args=None) -> Tuple[Any, Any]:
    # parse args.optim_args
    optim_args = {}
    if optim_args:
        for mapping in optim_args.replace(" ", "").split(","):
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


def get_decay_parameter_names(model) -> List[str]:
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters


def main():
    seedEverything()
    tx8_replace_func.replace_TrainingArguments_setup_devices()
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    if "--use_cuda" in sys.argv and torch.cuda.device_count() > 1:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    
   
    
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    # import torch
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config_all = 0
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        config_all = config
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        config_all = config
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        cconfig_all = config
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    config.text_config.num_hidden_layers = model_args.llama_num_hidden_layers
    config.vision_config.num_hidden_layers = model_args.clip_num_hidden_layers
    
    
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
    import inspect
    import importlib
    # print('!!!!!!!!!!!!!!!!!!!!!!') 
    # print(model)
    # print(inspect.getfile(model.__class__)) # 查看model的文件实现具体位置
    # print('!!!!!!!!!!!!!!!!!!!!!!') 
    is_fsdp_xla_enabled = training_args.fsdp_config["xla"]
    num_devices = pjrt.global_device_count()
    
    #替换融合算子以及函数
    if not model_args.use_cuda:
        tx8_replace_func.replace_merge_input_ids_with_image_features_with_new_func()
        tx8_replace_func.replace_forward_with_new_func()
        tx8_replace_func.replace_LlamaRotaryEmbedding_forward_with_new_func()
        if torch_xla._XLAC.IsCurrentDeviceTx8():
            tx8_replace_func.replace_tx8_mask()
            tx8_replace_func.replace_tx8_rmsNorm(model)
            tx8_replace_func.replace_tx8_rope()

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
    # print("##########################")
    # Simulate some user inputs
    if model_args.use_cuda:
        torch_device = 'cuda'
    else:
        torch_device = xm.xla_device() 
     

    # 加载模型和处理器
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    # 创建数据集和数据加载器
    dataset = SupervisedDataset(
        json_path=data_args.json_file_path,
        image_root=data_args.images_path,
        processor=processor,
        max_text_length=data_args.block_size,
        use_cuda = model_args.use_cuda,
    )

    model = model.to(torch_device)
    model.train()
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = get_optimizer_cls_and_kwargs(training_args, model_args)
    optimizer = optimizer_cls(optimizer_grouped_parameters,**optimizer_kwargs)
    #学习率调度器
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(training_args.num_train_epochs),
        num_training_steps=training_args.num_train_epochs,
        scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
    )
   
    
    #！！！！！以下注释内容暂未适配
    # train_dataset = SupervisedDataset()
    # def get_train_dataloader(train_dataset) -> DataLoader:
    #     if train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
    #     dataloader_params = {
    #         "batch_size": 1,
    #     }

    #     return DataLoader(train_dataset, **dataloader_params)
    # def _xla_sharded_dataloader(dataloader):
    #     if is_torch_xla_available():
    #         import torch_xla.experimental.xla_sharding as xs
    #         import torch_xla.distributed.parallel_loader as pl
    #         sharding_spec = xs.ShardingSpec(spmd_mesh, (0, None)) 
    #         # TODO(jonbolin): Once integrated with Accelerate, we can use the Accelerate-prepared
    #         # MpDeviceLoader instead of manually adding sharding and adding a dataset attribute.
    #         loader = pl.MpDeviceLoader(dataloader, training_args.device, input_sharding=sharding_spec, loader_prefetch_size=training_args.train_batch_size, device_prefetch_size=4)
    #         loader.dataset = dataloader.dataset
    #         return loader
    #     else:
    #         return dataloader
    # #train_dataloader = _xla_sharded_dataloader(get_train_dataloader(train_dataset))
    # train_dataloader = get_train_dataloader(train_dataset)
    # Make sure that the loss is properly computed


    try:
        rank = dist.get_rank()
    except:
        rank=0
    from torch.utils.tensorboard import SummaryWriter
    #loss曲线保存路径
    writer = SummaryWriter(log_dir=f"{training_args.output_dir}/loss/rank{rank}")  
    tr_loss = 0
    global_step = 0
    for step, inputs in enumerate(dataset):
        optimizer.zero_grad()  #清除梯度           
        outputs = model(       #前向
            pixel_values=inputs['pixel_values'].squeeze(1).to(torch.float),
            input_ids=inputs['input_ids'].squeeze(1).to(torch.long),
            attention_mask=inputs['attention_mask'].squeeze(1).to(torch.long),
            labels=inputs['labels'].to(torch.long)
        ) 

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]     #loss   
        loss.backward() #反向
        
        # PrecisionDebuggerEND()
        
        optimizer.step()#更新参数
        lr_scheduler.step()              
        xm.mark_step()
        # PrecisionDebuggerMarkStep()
        
        #生成loss曲线
        tr_loss_step = loss.detach()
        tr_loss = tr_loss_step.to(torch.float)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar('train/learning_rate', current_lr, global_step)
        writer.add_scalar('train/loss', tr_loss, global_step)
        global_step +=1 
        
        print(f"!!!!!!!!!!!step: {step} , loss: {tr_loss:.5f}, learning rate: {current_lr}" ,flush=True)

        if global_step == 10:
            break


def _mp_fn(index):
    # For xla_spawn (TPUs)
    print(f"!!!!!!rank:{index},CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12465'
    dist.init_process_group('xla', init_method='pjrt://')
    main()

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional

import datasets
# import evaluate
import torch
from datasets import load_dataset
import torch.distributed as dist
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    # TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint, FSDPOption
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import tx8_arguments
import numpy as np
import random

#os.environ['TX8_MODEL_EXPORT_LEVEL']="11"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.1")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


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

import json
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["messages"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["target_ids"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


class MegatronIndexedDataset(Dataset):
    def __init__(
        self, path_prefix, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        from indexed_dataset import IndexedDataset
        super(MegatronIndexedDataset, self).__init__()

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.cached_data_dict = {}
        self.indexed_dataset = IndexedDataset(path_prefix)
        print("MegatronIndexedDataset init done...")

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        input_ids = torch.tensor(self.indexed_dataset[i], dtype=torch.int)[:self.max_len]
        if input_ids.size(0) < self.max_len:
            padding_length = self.max_len - input_ids.size(0)
            input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=self.tokenizer.pad_token_id)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        MegatronIndexedDataset if data_args.lazy_preprocess else None
    )
    print("Loading data...")

    dataset_cls = MegatronIndexedDataset
    train_dataset = dataset_cls(data_args.data_path, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_data = []
        with open(data_args.eval_data_path, "r") as f:
            for line in f:
                eval_data.append(json.loads(line))
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset)

def make_supervised_data_module_flagscale(data_args, tokenizer_path, max_len):
    # This is for the FlagScale dataset(Megatron backend) from ZhiYuan, need wudao/pile-cc bin&idx files prefix path.
    from megatron_fs_dataset import FlagscaleMegatronDataset
    train_dataset = FlagscaleMegatronDataset(data_args.data_path, tokenizer_path=tokenizer_path, max_len=max_len)

    return dict(train_dataset=train_dataset)


def main(mg_semap=None):
    # 为了与cuda程序兼容，将xla的导入放到这里
    if "--use_cuda" not in sys.argv:
        import tx8_replace_func
        import tx8_util
        import torch_xla
        import torch_xla.core.xla_model as xm
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    if "--use_cuda" in sys.argv and torch.cuda.device_count() > 1:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)

    seedEverything()

    parser = HfArgumentParser((tx8_arguments.ModelArguments, tx8_arguments.DataTrainingArguments, transformers.TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: " + str("cuda" if model_args.use_cuda else training_args.device) + f", n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )    
    logger.info(f"Training/evaluation parameters {training_args}")
    if not model_args.use_cuda:
        tx8_replace_func.replace_tx8_mixtral_rotary_embedding()
    # Detecting last checkpoint.
    last_checkpoint = None

    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    # set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.


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
    # chagne layer number
    config.max_position_embeddings = data_args.block_size
    config.num_hidden_layers = model_args.num_hidden_layers
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        #tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        #model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,torch_dtype=torch_dtype,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            attn_implementation="eager" if model_args.use_cuda else None,
        )
    else:
        from torchdistx import deferred_init
        model = deferred_init.deferred_init(AutoModelForCausalLM.from_config, config)
        # model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        # n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        # logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if not model_args.use_cuda:
        if model_args.use_flash_attn:
            model = tx8_replace_func.enable_tx8_flash_attention(model)
        tx8_replace_func.replace_TrainingArguments_setup_devices()
        tx8_replace_func.replace_tx8_rope(config)
        tx8_replace_func.replace_tx8_one_hot()
        tx8_replace_func.replace_tx8_rmsNorm(model)
        tx8_replace_func.replace_tx8_mask()
        model = tx8_replace_func.replace_MixtralSparseMoeBlock_customcall(model)
        model = tx8_replace_func.replace_MixtralBlockSparseTop2MLP_customcall(model)
        # 打印模型中的module_name 和 参数的数据类型
        # for module_name, m in model.named_modules():
        #     for n, p in m.named_parameters(recurse=False):
        #         print(f"module_name={module_name}  p.dtype={p.dtype}")

    is_fsdp_xla_enabled = training_args.fsdp_config["xla"]
    if is_fsdp_xla_enabled and not model_args.use_cuda:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp import checkpoint_module
        from torch_xla.distributed.fsdp.wrap import size_based_auto_wrap_policy,transformer_auto_wrap_policy
        import functools
        from transformers.trainer_pt_utils import get_module_class_from_name

        auto_wrap_policy = None
        auto_wrapper_callable = None
        default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = training_args.fsdp_config.get("fsdp_transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap)

        if training_args.fsdp_config["min_num_params"] > 0:
            auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=training_args.fsdp_config["min_num_params"])
        elif fsdp_transformer_layer_cls_to_wrap is not None:
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
        if training_args.fsdp_config["xla_fsdp_grad_ckpt"]:
            if model.config.use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                model.config.use_cache = False
            # Apply gradient checkpointing to auto-wrapped sub-modules if specified
            def auto_wrapper_callable(m, *args, **kwargs):
                return FSDP(checkpoint_module(m), *args, **kwargs)
        
        fsdp_kwargs = training_args.xla_fsdp_config
        _init_with_torchdistX = None if model_args.model_name_or_path else functools.partial(
            tx8_util._init_with_torchdistX,split_weight_path = model_args.deferred_init_model_path,num_hidden_layers=config.num_hidden_layers)
        if model_args.fsdp_dp_sharding > 1  and model_args.megatron_tp_sharding > 1:
            assert model_args.fsdp_dp_sharding * model_args.megatron_tp_sharding == dist.get_world_size() , \
            "fsdp_dp_sharding * megatron_tp_sharding shoud equal to global_device_count"
            tx8_replace_func.replace_tx8_mixtral_decode_forward()
            for decoder_layer in model.model.layers:
                decoder_layer.self_attn.num_heads = int(decoder_layer.self_attn.num_heads/model_args.megatron_tp_sharding)
                decoder_layer.self_attn.num_key_value_heads = int(decoder_layer.self_attn.num_key_value_heads/model_args.megatron_tp_sharding)
                decoder_layer.self_attn.hidden_size = int(decoder_layer.self_attn.hidden_size/model_args.megatron_tp_sharding)
            fsdp_tp_kwargs = fsdp_kwargs.copy()
            
            '''
            #4cards 2fsdp+2tp
            fsdp_tp_kwargs["sharding_rank"] = dist.get_rank() 
            fsdp_tp_kwargs["sharding_world_size"] = dist.get_world_size()
            fsdp_tp_kwargs["param_init_megatron_tp"] = model_args.megatron_tp_sharding
            fsdp_tp_kwargs["param_init_fsdp_dp"] = model_args.fsdp_dp_sharding
            all_reduce_sharding_groups = [[0,2],[1,3]]
            tx8_util.all_reduce_sharding_group_load(all_reduce_sharding_groups)
            fsdp_tp_kwargs["tp_param_gather_sharding_groups"] = [[0,1],[2,3]]
            fsdp_tp_kwargs["tp_reduce_scatter_sharding_groups"] = [[0,1],[2,3]]
            fsdp_tp_kwargs["ignore_params"] = True
            '''
            '''
            #8cards 4fsdp+2tp
            fsdp_kwargs["sharding_groups"] = [[0,1,2,3],[4,5,6,7]]
            fsdp_kwargs["sharding_rank"] = dist.get_rank() % model_args.fsdp_dp_sharding
            fsdp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding

            fsdp_tp_kwargs["sharding_rank"] = dist.get_rank() % model_args.fsdp_dp_sharding
            fsdp_tp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding 
            fsdp_tp_kwargs["param_init_megatron_tp"] = model_args.megatron_tp_sharding
            fsdp_tp_kwargs["param_init_fsdp_dp"] = model_args.fsdp_dp_sharding
            all_reduce_sharding_groups = [[0,2],[1,3],[4,6],[5,7]]
            tx8_util.all_reduce_sharding_group_load(all_reduce_sharding_groups)
            fsdp_tp_kwargs["tp_param_gather_sharding_groups"] = [[0,1],[2,3],[4,5],[6,7]]
            fsdp_tp_kwargs["tp_reduce_scatter_sharding_groups"] = [[0,1,4,5],[2,3,6,7]]
            fsdp_tp_kwargs["ignore_params"] = True
            '''
            total_rank = model_args.fsdp_dp_sharding * model_args.megatron_tp_sharding
            total_tensor = torch.tensor(range(total_rank))
            dp_tensor = total_tensor.view(-1, model_args.fsdp_dp_sharding)
            fsdp_tp_kwargs["ignore_params"] = ["input_layernorm","post_attention_layernorm","gate_moe"]
            if training_args.fsdp[0] == FSDPOption.FULL_SHARD:
                fsdp_tp_kwargs["sharding_rank"] = dist.get_rank() 
                fsdp_tp_kwargs["sharding_world_size"] = dist.get_world_size()
                fsdp_tp_kwargs["param_init_megatron_tp"] = model_args.megatron_tp_sharding
                fsdp_tp_kwargs["param_init_fsdp_dp"] = model_args.fsdp_dp_sharding
                fsdp_tp_kwargs["tp_param_gather_sharding_groups"] = dp_tensor.tolist()
                fsdp_tp_kwargs["tp_reduce_scatter_sharding_groups"] = dp_tensor.tolist()
                all_reduce_sharding_groups = dp_tensor.transpose(0, 1).tolist()
                model_args.all_reduce_sharding_groups = all_reduce_sharding_groups
                tx8_util.all_reduce_sharding_group_load(all_reduce_sharding_groups)
            else:
                fsdp_kwargs["sharding_groups"] = dp_tensor.tolist()
                reduce_scatter_list = []
                first_group = []
                for index in range(int(model_args.fsdp_dp_sharding  / model_args.megatron_tp_sharding)):
                    item = [val*(model_args.megatron_tp_sharding+1) for val in range(model_args.megatron_tp_sharding)] if index == 0 \
                        else [first_group[-1] + 1 + val for val in first_group[0:model_args.megatron_tp_sharding]]
                    first_group.extend(item)
                reduce_scatter_list.append(first_group)
                for index in range(model_args.megatron_tp_sharding-1):
                    item = [ (val + model_args.megatron_tp_sharding) % (model_args.fsdp_dp_sharding  * model_args.megatron_tp_sharding)  for val in reduce_scatter_list[-1]]
                    reduce_scatter_list.append(item)
                print(f"!!!!!!!{reduce_scatter_list=}")
                fsdp_kwargs["tp_reduce_scatter_sharding_groups"] = reduce_scatter_list
                fsdp_kwargs["sharding_rank"] = dist.get_rank() % model_args.fsdp_dp_sharding
                fsdp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding

                fsdp_tp_kwargs["sharding_rank"] = dist.get_rank() % model_args.fsdp_dp_sharding
                fsdp_tp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding 
            fsdp_tp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding 
                fsdp_tp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding 
                fsdp_tp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding 
                fsdp_tp_kwargs["sharding_world_size"] = model_args.fsdp_dp_sharding 
                fsdp_tp_kwargs["param_init_megatron_tp"] = model_args.megatron_tp_sharding
                fsdp_tp_kwargs["param_init_fsdp_dp"] = model_args.fsdp_dp_sharding

                all_reduce_sharding_groups = []
                tp_param_gather_sharding_groups = []
                for i in range(dp_tensor.size()[0]):
                    dp_i = dp_tensor[i]
                    all_gather = dp_i.view(model_args.megatron_tp_sharding, -1)
                    all_reduce = all_gather.transpose(0, 1)
                    tp_param_gather_sharding_groups.extend(all_gather.tolist())
                    all_reduce_sharding_groups.extend(all_reduce.tolist())
                model_args.all_reduce_sharding_groups = all_reduce_sharding_groups
                tx8_util.all_reduce_sharding_group_load(all_reduce_sharding_groups)
                fsdp_tp_kwargs["tp_param_gather_sharding_groups"] = tp_param_gather_sharding_groups
                fsdp_tp_kwargs["tp_reduce_scatter_sharding_groups"] = torch.tensor(all_reduce_sharding_groups).transpose(0, 1).tolist()
        
            def auto_wrapper_callable(m, *args, **kwargs):
                return FSDP(m, param_init_fn=_init_with_torchdistX, *args, **fsdp_tp_kwargs)

        fsdp_wrap = lambda m: FSDP(m,auto_wrap_policy=auto_wrap_policy,auto_wrapper_callable=auto_wrapper_callable,param_init_fn=_init_with_torchdistX,**fsdp_kwargs,)
        import inspect
        forward_signature = inspect.signature(model.forward.__func__)
        model = fsdp_wrap(model)
        model.forward.__func__.__signature__ = forward_signature

    elif is_fsdp_xla_enabled and model_args.use_cuda:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap,
        )
        from torch.distributed.fsdp.api import ShardingStrategy
        import functools
        from transformers.trainer_pt_utils import get_module_class_from_name
        transformer_cls_to_wrap = set()
        default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = training_args.fsdp_config.get("transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap)
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
        # if model_args.fsdp_dp_sharding > 1  and model_args.megatron_tp_sharding > 1:
        from torch.distributed.device_mesh import init_device_mesh
        data_parallel_size = dist.get_world_size() if model_args.fsdp_dp_sharding == -1 else model_args.fsdp_dp_sharding
        tensor_parallel_size = model_args.megatron_tp_sharding
        device_mesh = init_device_mesh(device_type="cuda",
                                    mesh_shape=(data_parallel_size, tensor_parallel_size),
                                    mesh_dim_names=("data_parallel", "tensor_parallel"),
                                    )
        dp_mesh = device_mesh["data_parallel"]
        tp_mesh = device_mesh["tensor_parallel"]        
        if data_parallel_size > 1  and tensor_parallel_size > 1:
            from torch.distributed._tensor import DeviceMesh
            from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
            for layer in model.model.layers:
                layer_parallelize_plan = {} 
                layer_parallelize_plan["self_attn.q_proj"] = ColwiseParallel()
                layer_parallelize_plan["self_attn.k_proj"] = ColwiseParallel()
                layer_parallelize_plan["self_attn.v_proj"] = ColwiseParallel()
                layer_parallelize_plan["self_attn.o_proj"] = RowwiseParallel()

                layer_parallelize_plan["mlp.gate_proj"] = ColwiseParallel()
                layer_parallelize_plan["mlp.up_proj"] = ColwiseParallel()
                layer_parallelize_plan["mlp.down_proj"] = RowwiseParallel()

                parallelize_module(layer, tp_mesh, layer_parallelize_plan)

            for layer in model.model.layers:
                assert model.model.config.num_attention_heads % tensor_parallel_size == 0
                layer.self_attn.num_heads = model.model.config.num_attention_heads // tensor_parallel_size
                layer.self_attn.num_key_value_heads = model.model.config.num_key_value_heads // tensor_parallel_size
                layer.self_attn.hidden_size = model.model.config.hidden_size // tensor_parallel_size

        import inspect
        forward_signature = inspect.signature(model.forward.__func__)
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_mesh=dp_mesh,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True)
        model.forward.__func__.__signature__ = forward_signature
    
    print(f"!!!!model: {model}")

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    from transformers.testing_utils import CaptureLogger
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
            # output = tokenizer(examples[text_column_name], padding='max_length', max_length = data_args.block_size)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    # if hasattr(config, "max_position_embeddings"):
    #     max_pos_embeddings = config.max_position_embeddings
    # else:
    #     # Define a default value if the attribute is missing in the config.
    #     max_pos_embeddings = 1024
    max_pos_embeddings = 1024
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    data_args.lazy_preprocess = True
    data_args.data_path = data_args.dataset_name
    data_args.eval_data_path = None
    if data_args.use_flagscale_dataset:
        data_module = make_supervised_data_module_flagscale(
            data_args, model_args.model_name_or_path, max_len=data_args.block_size
        )
    else:
        data_module = make_supervised_data_module(
            tokenizer=tokenizer, data_args=data_args, max_len=data_args.block_size
        )

    # Initialize our Trainer
    from xla_trainer_fsdp_tp import XlaTrainer
    trainer = XlaTrainer(
        model=model,
        args=training_args,
        model_args=model_args,
        eval_dataset=None,
        tokenizer=tokenizer,
        tsprobe_config = model_args.tsprobe_config,
        dump_path = model_args.dump_path,   
        gloal_semaphore = mg_semap,     
        **data_module,
    )


    # Training
    if training_args.do_train:
        train_result = trainer.train()

    print("*****************do training doine*************")
    exit(0)




def _mp_fn(local_rank, mg_semap):
    import torch_xla.experimental.pjrt_backend
    dist.init_process_group('xla', init_method='pjrt://')
    # if local_rank != 0:
    #     sys.stdout = open(os.devnull, "w")
    main(mg_semap)

def _mp_device_preprocess(global_rank, group_rank, local_rank, fsdp_dp_sharding=-1, megatron_tp_sharding=-1):
    import torch_xla.core.xla_env_vars as xenv
    import torch_xla.core.xla_model as xm
    if fsdp_dp_sharding > 1  and megatron_tp_sharding > 1 and megatron_tp_sharding % 4 == 0:
        def func():
            global_world_size = fsdp_dp_sharding * megatron_tp_sharding
            dp_tensor = torch.tensor(range(global_world_size)).view(-1, fsdp_dp_sharding)
            host_count = dp_tensor.size()[0]
            host_all_reduce = []
            for i in range(host_count):
                all_reduce = dp_tensor[i].view(megatron_tp_sharding, -1).transpose(0, 1)
                host_all_reduce.append(all_reduce)
            device_distributions = None
            for i in list([0, 2, 4, 6, 16, 18, 20, 22]):
                mul_device = torch.tensor([[i,i+8,i+1,i+9]])
                if device_distributions is None:
                    device_distributions = mul_device
                else:
                    device_distributions = torch.cat([device_distributions,mul_device],dim=0)
            
            for all_reduce in host_all_reduce:    
                for i, row in enumerate(all_reduce):
                    for j, value in enumerate(row):
                        if value == global_rank:
                            xm.set_tx8_device(device_distributions[i,j])
                            return
        func()
    else:
        xm.set_tx8_device(local_rank)
    print(f"!!!!!!_mp_device_preprocess local_rank:{global_rank},TX8_ACTIVE_DEVICE_ID: {os.environ['TX8_ACTIVE_DEVICE_ID']}")

if __name__ == "__main__":
    main()

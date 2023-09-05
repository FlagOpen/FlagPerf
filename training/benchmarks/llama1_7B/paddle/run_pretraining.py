"""LLaMA Pretraining"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import sys
import time

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
    speed_metrics,
)

from paddlenlp.transformers import (
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    LlamaConfig,
    LlamaForCausalLM,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, PaddleCallback, dist_paddle
from driver.config_manager import get_properties_from_config
from dataloaders.dataloader import get_train_data_file
from dataloaders.dataloader import create_pretrained_dataset
from train.trainer import PretrainingTrainer
from train.training_state import TrainingState

MODEL_CLASSES = {
    "llama": (
        LlamaConfig,
        LlamaForCausalLM,
    ),
}

def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
    min_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate deacyed to."},
    )
    decay_steps: float = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, will use the min_learning_rate."
        },
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluating.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    input_dir: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split."})

    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    share_folder: bool = field(
        default=False,
        metadata={"help": "Use share folder for data dir and output dir on multi machine."},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(
        default="llama", metadata={"help": "Only support for llama pre-training for now."}
    )
    model_name_or_path: str = field(
        default="facebook/tiny-random-llama",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "use_flash_attention"},
    )
    use_fused_rms_norm: bool = field(
        default=False,
        metadata={"help": "llama, use_fused_rms_norm"},
    )
    fuse_attention_qkv: bool = field(
        default=True,
        metadata={"help": "gpt, fuse_attention_qkv"},
    )
    recompute_granularity: str = field(
        default="full",
        metadata={"help": "full core_attn"},
    )
    virtual_pp_degree: int = field(
        default=1,
        metadata={"help": "virtual_pp_degree"},
    )

    continue_training: bool = field(
        default=False,
        metadata={
            "help": "Pre-training from existing paddlenlp model weights. Default Fasle and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models."
        },
    )

def main():
    import config
    from config import mutable_params
    llama_driver = Driver(config, mutable_params)
    llama_driver.setup_config(argparse.ArgumentParser("Llama"))
    llama_driver.setup_modules(globals(), locals())
    training_state = TrainingState()

    dist_paddle.barrier()
    llama_driver.event(Event.INIT_START)
    init_start_time = llama_driver.logger.previous_log_time

    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(
        get_properties_from_config(config)
    )
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
    
    dist_paddle.set_seed(training_args)
    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    training_args.eval_iters = 10
    training_args.test_iters = training_args.eval_iters * 10

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # if last_checkpoint is None and len(
        #         os.listdir(training_args.output_dir)) > 1:
        #     raise ValueError(
        #         f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #         "Use --overwrite_output_dir to overcome.")
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    config_class, model_class = MODEL_CLASSES[model_args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    llama_config = config_class.from_pretrained(model_args.model_name_or_path)
    llama_config.max_position_embeddings = max(llama_config.max_position_embeddings, data_args.max_seq_length)
    if not model_args.continue_training:
        llama_config.vocab_size = max(llama_config.vocab_size, ((tokenizer.vocab_size - 1) // 128 + 1) * 128)
        logger.info(f"Reset vocab size to {llama_config.vocab_size} for batter amp peformance.")

    llama_config.lm_shift_labels = False
    llama_config.use_flash_attention = model_args.use_flash_attention
    llama_config.use_fused_rms_norm = model_args.use_fused_rms_norm
    llama_config.fuse_attention_qkv = False
    llama_config.recompute_granularity = model_args.recompute_granularity
    llama_config.virtual_pp_degree = model_args.virtual_pp_degree
    llama_config.use_recompute = training_args.recompute

    llama_config.tensor_parallel_degree = training_args.tensor_parallel_degree
    llama_config.tensor_parallel_rank = training_args.tensor_parallel_rank

    print("Final pre-training config:", llama_config)

    dtype = "float32"
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    if training_args.pipeline_parallel_degree > 1 and model_args.model_type == "llama":
        model_class = LlamaForCausalLMPipe

    if model_args.continue_training:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=llama_config,
            dtype=dtype,
            load_state_as_np=True,
        )
    else:
        model = model_class._from_config(llama_config, dtype=dtype)

    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps
    warmup_steps = training_args.warmup_ratio * training_args.max_steps

    lr_scheduler = None
    if training_args.lr_scheduler_type.value == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    elif training_args.lr_scheduler_type.value == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    
    data_file = get_train_data_file(data_args)
    train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
        data_args, training_args, data_file, tokenizer
    )

    print(f"train_dataset length:{len(train_dataset)}")
    print(f"eval_dataset length:{len(eval_dataset)}")

    trainer = PretrainingTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(None, lr_scheduler),
        tokenizer=tokenizer,
        callbacks=[PaddleCallback(driver=llama_driver)],
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if not config.do_train:
        return config, training_state

    llama_driver.event(Event.INIT_END)
    init_end_time = llama_driver.logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    dist_paddle.barrier()

    train_start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    training_state.raw_train_time = time.time() - train_start_time
    
    return config, training_state

if __name__ == "__main__":
    now = time.time()

    config, state = main()

    if not dist_paddle.is_main_process():
        exit()

    e2e_time = time.time() - now
    training_perf = (dist_paddle.global_batch_size(config) *
                     state.global_steps) / state.raw_train_time
    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_avg_loss,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

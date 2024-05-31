"""GPT Pretraining"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import paddlenlp
import paddle
from dataloaders.dataset import create_pretrained_dataset, get_train_data_file
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, set_seed
from paddlenlp.transformers import (
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    GPTConfig,
    GPTForCausalLM,
    GPTForCausalLMPipe,
    LinearAnnealingWithWarmupDecay,
)
from paddlenlp.metrics import Perplexity
from paddlenlp.trainer import EvalPrediction
from paddlenlp.utils.log import logger
from train.trainer import PretrainingTrainer
from train.training_state import TrainingState

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import Driver, Event, dist_paddle
from driver.config_manager import get_properties_from_config

MODEL_CLASSES = {
    "gpt": (
        GPTConfig,
        GPTForCausalLM,
    ),
}

def ppl(x: EvalPrediction):
    perplexity = Perplexity()
    predictions = paddle.to_tensor(x.predictions)
    labels = paddle.to_tensor(x.label_ids)
    correct = perplexity.compute(predictions, labels)
    perplexity.update(correct.numpy())
    ret = perplexity.accumulate()
    return {"ppl":ret}
    

@dataclass
class PreTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="output",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
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
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    split: str = field(
        default="949,50,1", metadata={"help": "Train/valid/test data split."}
    )

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

    data_impl: str = field(default="mmap", metadata={"help": "The format of the preprocessed data."})
    skip_warmup: bool = field(
        default=True,
        metadata={"help": "Whether to skip the warmup process of mmap files."},
    )
    data_cache: str = field(default=None, metadata={"help": "The path of the cached dataset."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(default="gpt", metadata={"help": "Only support for gpt pre-training for now."})
    model_name_or_path: str = field(
        default="gpt2-medium-en",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    output_attentions: bool = field(default=False, metadata={"help": "Whether output attention weights"})
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    virtual_pp_degree: int = field(
        default=1,
        metadata={"help": "virtual_pp_degree"},
    )
    use_flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use flash attention"}
    )
    fused_linear: bool = field(
        default=False,
        metadata={"help": "gpt, whether to fuse linear projection"},
    )
    fuse_attention_qkv: bool = field(
        default=False,
        metadata={"help": "gpt, whether to fuse attention qkv"},
    )
    enable_fuse_transformer: bool = field(
        default=False,
        metadata={"help": "gpt, enable_fuse_transformer"},
    )
    hidden_dropout_prob: float = field(
        default=0.1, metadata={"help": "The hidden dropout prob."}
    )
    attention_probs_dropout_prob: float = field(
        default=0.1, metadata={"help": "The attention hidden dropout prob."}
    )
    continue_training: bool = field(
        default=True,
        metadata={
            "help": "Pre-training from existing paddlenlp model weights. Default False and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models."
        },
    )


def main():

    import config
    from config import mutable_params

    paddlenlp.transformers.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["gpt3-6.7B-en"] = 2048
    paddlenlp.transformers.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["gpt3-13B-en"] = 2048
    paddlenlp.transformers.GPT_PRETRAINED_INIT_CONFIGURATION["gpt3-6.7B-en"]["max_position_embeddings"] = 2048
    paddlenlp.transformers.GPT_PRETRAINED_INIT_CONFIGURATION["gpt3-13B-en"]["max_position_embeddings"] = 2048

    gpt_driver = Driver(config, mutable_params)
    gpt_driver.setup_config(argparse.ArgumentParser("gpt"))
    gpt_driver.setup_modules(globals(), locals())
    training_state = TrainingState()

    dist_paddle.barrier()
    gpt_driver.event(Event.INIT_START)
    init_start_time = gpt_driver.logger.previous_log_time

    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(
        get_properties_from_config(config)
    )
    data_args.input_dir = gpt_driver.config.data_dir

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    set_seed(seed=training_args.seed, args=training_args)
    # paddlenlp.trainer.set_seed(seed=training_args.seed, args=training_args)
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

    config_class, model_class = MODEL_CLASSES[model_args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    gpt_config = config_class.from_pretrained(model_args.model_name_or_path)
    gpt_config.output_attentions = model_args.output_attentions
    gpt_config.max_position_embeddings = max(
        gpt_config.max_position_embeddings, data_args.max_seq_length
    )
    gpt_config.virtual_pp_degree = model_args.virtual_pp_degree
    gpt_config.hidden_dropout_prob = model_args.hidden_dropout_prob
    gpt_config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
    gpt_config.enable_fuse_transformer = model_args.enable_fuse_transformer
    gpt_config.fuse_attention_qkv = model_args.fuse_attention_qkv
    gpt_config.use_recompute = training_args.recompute
    gpt_config.use_flash_attention = model_args.use_flash_attention

    gpt_config.tensor_parallel_degree = training_args.tensor_parallel_degree
    gpt_config.tensor_parallel_rank = training_args.tensor_parallel_rank

    print("Final pre-training config:", gpt_config)

    # Set the dtype for loading model
    dtype = "float32"
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    if training_args.pipeline_parallel_degree > 1 and model_args.model_type == "gpt":
        model_class = GPTForCausalLMPipe

    if model_args.continue_training:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=gpt_config,
            dtype=dtype,
            load_state_as_np=True,
        )
    else:
        model = model_class(gpt_config)

    if training_args.recompute:
        model.recompute_enable()
    # Create the learning_rate sheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    lr_scheduler = None
    if training_args.lr_scheduler_type.value == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=training_args.warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    elif training_args.lr_scheduler_type.value == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=training_args.warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )

    data_file = get_train_data_file(data_args)
    (
        train_dataset,
        eval_dataset,
        test_dataset,
        data_collator,
    ) = create_pretrained_dataset(data_args, training_args, data_file, tokenizer, need_data=training_args.should_load_dataset)
    
    total_effective_tokens = (
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps
        * data_args.max_seq_length
    )

    trainer = PretrainingTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(None, lr_scheduler),
        tokenizer=tokenizer,
        callbacks=[dist_paddle.PaddleCallback(driver=gpt_driver)],
        compute_metrics=ppl,
    )

    dist_paddle.barrier()
    gpt_driver.event(Event.INIT_END)
    init_end_time = gpt_driver.logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e3

    # Training
    dist_paddle.barrier()

    if training_args.do_train:
        train_result = trainer.train()
        train_metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        training_state.raw_train_time = train_metrics["train_runtime"]
        training_state.training_sequences_per_second = train_metrics["train_samples_per_second"]
        training_state.loss = train_metrics["train_loss"]
        training_state.effective_tokens_per_second = total_effective_tokens / train_metrics["train_runtime"]


    # Evaluation
    # dist_paddle.barrier()
    # eval_metrics = trainer.evaluate()
    # training_state.eval_loss = eval_metrics["eval_loss"]
    # if eval_metrics["eval_ppl"] < config.target_ppl:
    #     training_state.converged_success()
        
    return training_args, training_state, gpt_driver


if __name__ == "__main__":
    now = time.time()

    training_args, state, driver = main()

    if not dist_paddle.is_main_process():
        exit()

    e2e_time = time.time() - now
    if training_args.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": state.training_sequences_per_second,
            "effective_tokens_per_second": state.effective_tokens_per_second,
            # "converged": state.converged,
            # "final_loss": state.eval_loss,
            # "final_ppl": state.eval_ppl,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    driver.logger.log(Event.FINISHED, message=finished_info, stacklevel=0)

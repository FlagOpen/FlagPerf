from dataclasses import dataclass
from importlib import import_module
import logging
import pathlib
import torch
import transformers
import tokenizers
import argparse
from train.llava_trainer import LLaVATrainer
from utils import conversation as conversation_lib
from model import *
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

from config.arguments_pretrain import DataArguments_pretrain, ModelArguments_pretrain, TrainingArguments_pretrain
from config.arguments_finetune import DataArguments_finetune, ModelArguments_finetune, TrainingArguments_finetune
from utils.model_save import safe_save_model_for_hf_trainer
from dataset.llava_dataset import make_supervised_data_module
import os
import sys


class MyLogHandler(logging.Handler, object):

    def __init__(self):
        logging.Handler.__init__(self)
        self.texts = []

    def emit(self, record):
        msg = self.format(record)
        if 'train_samples_per_second' in msg:
            self.texts.append(msg)


def get_metric(texts):
    msg = texts[-1]
    meaningful_msg = msg.split('train_samples_per_second=')[1]
    pure_msg = meaningful_msg.split(',')[0]
    return float(pure_msg)


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Reserved for deepspeed framework")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--flagperf_config", type=str, default="/home/FlagPerf/training/nvidia/llava1.5_13b-deepspeed/config/config_A800x1x8.py")
    parser.add_argument("--node_rank",
                        type=int,
                        required=True,
                        help="The rank of the node for multi-node distributed training.")   
    parser.add_argument("--nnodes",
                        type=int,
                        required=True,
                        help="how many hosts to run the testcase.")
    parser.add_argument("--nproc_per_node",
                        type=int,
                        required=True,
                        help="how many processes will run on each host.")
    parser.add_argument("--deepspeed",type=str)
    return parser


def train(parser, remaining_argv=None, attn_implementation=None):
    global local_rank

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=remaining_argv)
    local_rank = training_args.local_rank

    bnb_model_from_pretrained_args = {}
    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainset = data_module[0]
    print(len(trainset))
    print(len(trainset[0]))
    trainset=trainset[:512]
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)

if __name__ == "__main__":

    arg_parser = get_argument_parser()
    args, remaining_argv = arg_parser.parse_known_args()

    sys.path.append(os.path.dirname(args.flagperf_config))
    config_file = os.path.basename(args.flagperf_config).split('.')[0]

    module = import_module(config_file)

    seqlength = getattr(module, 'seqlength')
    batchsize = getattr(module, 'batchsize')
    datafilename = getattr(module, 'datafilename')
    theoryflops = getattr(module, 'theoryflops')
    epochs = getattr(module, 'epochs')
    
    logger = logging.getLogger("hf_trainer")
    handler_pretrain = MyLogHandler()
    logger.addHandler(handler_pretrain)

    parser_pretrain = transformers.HfArgumentParser(
        (ModelArguments_pretrain, DataArguments_pretrain, TrainingArguments_pretrain))
    # train(parser_pretrain, attn_implementation="flash_attention_2")
    train(parser_pretrain, remaining_argv, attn_implementation="flash_attention_2")

    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 run_pretraining.py --nproc_per_node 8 --nnodes 1 --node_rank 0 --deepspeed ./zero2.json

    # # TODO stage-2: finetune
    # # finetune需要pretrain阶段生成的权重来进行指令微调->指定第一阶段生成的权重到特定位置(40mb左右的权重)
    # parser_finetune = transformers.HfArgumentParser(
    #     (ModelArguments_finetune, DataArguments_finetune, TrainingArguments_finetune))
    # train(parser_finetune, remaining_argv, attn_implementation="flash_attention_2")

    if local_rank == 0:
        tokens = seqlength * batchsize
        perf = get_metric(handler_pretrain.texts)
        whole_tps = tokens * perf
        chip_tps = whole_tps / args.nproc_per_node * args.nnodes
        print("Pretrain stage")
        print("System tokens per second: ", whole_tps)
        print("Tokens/p/s: ", chip_tps)
        TFLOPS = int(theoryflops/1000000000000)
        print("Theory TFLOPS: ", TFLOPS)
        print("Tokens/TFLOPS: ", chip_tps / TFLOPS)
        print("MFU: ", chip_tps * 13000000000.0 * 6 / theoryflops)

        # print("finetune stage")
        # print("System tokens per second: ", whole_tps)
        # print("Tokens/p/s: ", chip_tps)
        # TFLOPS = int(theoryflops/1000000000000)
        # print("Theory TFLOPS: ", TFLOPS)
        # print("Tokens/TFLOPS: ", chip_tps / TFLOPS)
        # print("MFU: ", chip_tps * 13000000000.0 * 6 / theoryflops)

        # print("two stage")
        # print("System tokens per second: ", whole_tps)
        # print("Tokens/p/s: ", chip_tps)
        # TFLOPS = int(theoryflops/1000000000000)
        # print("Theory TFLOPS: ", TFLOPS)
        # print("Tokens/TFLOPS: ", chip_tps / TFLOPS)
        # print("MFU: ", chip_tps * 13000000000.0 * 6 / theoryflops)


    # TODO 通过trainer TrainingArguments等来指定pretrain和finetune阶段不同的参数
    #      MFU的计算：stage1 MFU stage2 MFU all:MFU 计算三个指标
    # TODO eval: mmmu
    # eval_mmlu_llava()



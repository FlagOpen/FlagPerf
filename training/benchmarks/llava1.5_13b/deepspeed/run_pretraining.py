from importlib import import_module
import pathlib
import time
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

from evaluate.evaluator import eval_mmlu_llava

import os
import sys

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


def train(model_args, data_args, training_args, attn_implementation=None):
    global local_rank
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
    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 run_pretraining.py --nproc_per_node 8 --nnodes 1 --node_rank 0 --deepspeed ./zero2.json
    arg_parser = get_argument_parser()
    args, remaining_argv = arg_parser.parse_known_args()

    sys.path.append(os.path.dirname(args.flagperf_config))
    config_file = os.path.basename(args.flagperf_config).split('.')[0]
    module = import_module(config_file)
    theoryflops = getattr(module, 'theoryflops')

    # stage-1 pretrain
    # args.deepspeed = "ds_config_pretrain.json"
    # parser_pretrain = transformers.HfArgumentParser(
    #     (ModelArguments_pretrain, DataArguments_pretrain, TrainingArguments_pretrain))
    # model_args, data_args, training_args = parser_pretrain.parse_args_into_dataclasses(args=remaining_argv)
    # model_args.model_name_or_path = getattr(module, 'model_name_or_path')
    # model_args.vision_tower = getattr(module, 'vision_tower')
    # data_args.data_path = getattr(module, 'pretrain_data_path')
    # data_args.image_folder = getattr(module, 'pretrain_image_folder')
    
    # start_time_pretrain = time.time()
    # train(model_args, data_args, training_args, attn_implementation="flash_attention_2")
    # end_time_pretrain = time.time()

    # pretrain_time = end_time_pretrain - start_time_pretrain

    # stage-2: finetune
    # finetune需要pretrain阶段生成的权重来进行指令微调->指定第一阶段生成的权重到特定位置(40mb左右的权重)
    args.deepspeed = "ds_config_finetune.json"
    parser_finetune = transformers.HfArgumentParser(
        (ModelArguments_finetune, DataArguments_finetune, TrainingArguments_finetune))
    model_args, data_args, training_args = parser_finetune.parse_args_into_dataclasses(args=remaining_argv)
    model_args.model_name_or_path = getattr(module, 'model_name_or_path')
    model_args.vision_tower = getattr(module, 'vision_tower')
    model_args.pretrain_mm_mlp_adapter = getattr(module, 'pretrain_mm_mlp_adapter')
    data_args.data_path = getattr(module, 'finetune_data_path')
    data_args.image_folder = getattr(module, 'finetune_image_folder')
    training_args.output_dir = getattr(module, 'output_dir_finetune')
    start_time_finetune = time.time()
    train(model_args, data_args, training_args, attn_implementation="flash_attention_2")
    end_time_finetune = time.time()
    finetune_time = end_time_finetune - start_time_finetune

    # evaluate model
    mmmu_model_path = getattr(module, 'output_dir_finetune')
    mmmu_data_path = getattr(module, 'mmmu_data_path')
    mmmu_config_path = getattr(module, 'mmmu_config_path')
    mmmu_output_path = getattr(module, 'mmmu_output_path')
    mmmu = eval_mmlu_llava(mmmu_model_path, mmmu_data_path, mmmu_config_path, mmmu_output_path)

    if local_rank == 0:
        # print(pretrain_time)
        # tokens_pretrain = 606.9 # awk '{ total += $1; count++ } END { print total/count }' shape.txt
        # whole_tps_pretrain = (tokens_pretrain * 558128) / pretrain_time
        # chip_tps_pretrain = whole_tps_pretrain / args.nproc_per_node * args.nnodes
        # print("Pretrain stage")
        # print("System tokens per second: ", whole_tps_pretrain)
        # print("Tokens/p/s: ", chip_tps_pretrain)
        # TFLOPS = int(theoryflops/1000000000000)
        # print("Theory TFLOPS: ", TFLOPS)
        # print("Tokens/TFLOPS: ", chip_tps_pretrain / TFLOPS)
        # print("MFU: ", chip_tps_pretrain * 13000000000.0 * 2 / theoryflops)


        print(finetune_time)
        tokens_finetune = 1150
        whole_tps_finetune = (tokens_finetune * 665000) / finetune_time
        chip_tps_finetune = whole_tps_finetune / args.nproc_per_node * args.nnodes
        print("Finetune stage")
        print("System tokens per second: ", whole_tps_finetune)
        print("Tokens/p/s: ", chip_tps_finetune)
        TFLOPS = int(theoryflops/1000000000000)
        print("Theory TFLOPS: ", TFLOPS)
        print("Tokens/TFLOPS: ", chip_tps_finetune / TFLOPS)
        print("MFU: ", chip_tps_finetune * 13000000000.0 * 6 / theoryflops)

        # total_time = pretrain_time + finetune_time
        # mfu_average = (tokens_pretrain * 558128 * 13000000000.0 * 2 + 
        #                tokens_finetune * 665000 * 13000000000.0 * 6) / total_time / (args.nproc_per_node * args.nnodes) / TFLOPS
        # print("two-stage average")
        # print("MFU: ", mfu_average)
        # print("Actual computing power: ", mfu_average * TFLOPS)
        
        # print("MMMU: ", mmmu)





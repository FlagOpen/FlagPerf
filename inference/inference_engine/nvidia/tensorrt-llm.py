import os
import subprocess
import time
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from tensorrt_llm.builder import get_engine_version
import torch

import json
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, T5Tokenizer

def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        engine_config = json.load(f)

    if engine_version is None:
        return engine_config['builder_config']['name'], None

    model_arch = engine_config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = engine_config['pretrained_config']['chatglm_version']
    if model_arch == 'QWenForCausalLM':
        model_version = engine_config['pretrained_config']['qwen_type']
    return model_arch, model_version

def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    elif model_name == 'GemmaForCausalLM' or model_name == 'RecurrentGemmaForCausalLM':
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file,
                                   padding_side='left',
                                   truncation_side='left',
                                   legacy=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left',
                                legacy=False)

    if model_name == 'QWenForCausalLM' and model_version == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config['pad_token_id']
        end_id = gen_config['eos_token_id']
    elif model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id

class InferModel:
    def __init__(self, config, onnx_path, model):
        self.config = config

        if config.exist_tllm_checkpoint is None:
            tllm_checkpoint_path = self.convert_checkpoint(config)
        else:
            tllm_checkpoint_path = os.path.join(config.data_dir, config.exist_tllm_checkpoint)
        

        self.engine_path = self.build_engine(config, tllm_checkpoint_path)


    def convert_checkpoint(self, config):
        dir_path = os.path.join(config.perf_dir, "benchmarks", config.case, "tensorrt-llm")
        tllm_checkpoint = os.path.join(
            config.data_dir,
            "tllm_checkpoint_" + str(config.num_gpus) + "gpu" + 
            "_tp" + str(config.tp_size) + "_pp" + str(config.pp_size)
        )

        convert_cmd = "cd " + dir_path + " && "
        convert_cmd = convert_cmd + "python convert_checkpoint.py --model_dir " + config.weight_dir
        convert_cmd = convert_cmd + " --output_dir " + tllm_checkpoint
        convert_cmd = convert_cmd + " --dtype " + config.dtype
        if config.num_gpus != 1:
            convert_cmd = convert_cmd + " --tp_size " + config.tp_size
            convert_cmd = convert_cmd + " --pp_size " + config.pp_size
        
        p = subprocess.Popen(convert_cmd, shell=True)
        p.wait()

        return tllm_checkpoint
        
    def build_engine(self, config, tllm_checkpoint_path):
        if config.exist_compiler_path is None:
            tllm_path = os.path.join(config.data_dir, config.tllm_tmp_path)

            dir_tllm_path = os.path.dirname(tllm_path)
            os.makedirs(dir_tllm_path, exist_ok=True)

            time.sleep(10)

            build_engine_cmd = "cd " + config.dir_tllm_path + " && "
            build_engine_cmd = build_engine_cmd + "trtllm-build"
            build_engine_cmd = build_engine_cmd + " --checkpoint_dir " + tllm_checkpoint_path
            build_engine_cmd = build_engine_cmd + " --output_dir " + tllm_path
            build_engine_cmd = build_engine_cmd + " --gemm_plugin " + config.dtype

            p = subprocess.Popen(build_engine_cmd, shell=True)
            p.wait()
            return tllm_path
        else:
            return config.exist_compiler_path

    def __call__(self, model_inputs: list):
        start = time()
        output_len = 2
        input_lengths = [x.size(0) for x in model_inputs]


        engine_dir = self.engine_path
        model_name, model_version = read_model_name(engine_dir)
        tokenizer_path = os.path.join(self.config.data_dir, self.config.weight_dir)
        tokenizer, pad_id, end_id = load_tokenizer(
            tokenizer_dir = tokenizer_path,
            model_name = model_name,
            model_version = model_version
        )
        
        runtime_rank = tensorrt_llm.mpi_rank()
        model = ModelRunner.from_dir(engine_dir, rank = runtime_rank)

        compile_foo_time = time.time() - start
        outputs = model.generate(model_inputs,
                                 max_new_tokens = output_len,
                                 end_id = end_id,
                                 pad_id = pad_id)
        torch.cuda.synchronize()
        output_ids = outputs[0, 0, input_lengths[0]:]

        return output_ids, compile_foo_time

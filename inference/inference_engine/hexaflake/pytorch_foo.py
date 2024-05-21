import onnx
import onnxruntime
import torch
import os
import subprocess
import numpy as np
import time
import shutil
import argparse
import json
import hxrt as rt
import pickle as pk
from transformers import AutoTokenizer
from copy import deepcopy
from loguru import logger

from .hx_infexec import GLMInfer


class Graph(object):
    def __init__(
        self,
        engine,
        tokenizer_path,
        static_batch_size,
        batch_size,
        input_index,
        max_batch_size,
        base_length,
        constant_output=False,
        dev_ids = [],
        dev_dram_limit = [],
        dump_golden = [],
        split_strategy = [],
    ):
        global engine_version
        self.static_batch_size = static_batch_size
        self.dynamic_batch_size = batch_size
        self.base_length = base_length
        self.engine = engine

        self.model = GLMInfer(
            engine,
            batch_count=1,
            static_batch_size=static_batch_size,
            batch_size=batch_size,
            in_out_nparray=True,
            use_cache=True,
            max_batch_size=max_batch_size,
            dev_ids=dev_ids,
            dev_dram_limit=dev_dram_limit,
            dump_golden=dump_golden,
            split_strategy=split_strategy,
            config_file=os.path.join(tokenizer_path, "config.json"),
            base_length=base_length,
            engine_version=engine_version,
            constant_output=constant_output,
            total=total,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_index = input_index

   
    def graph_run(self, input_data, prompt, max_new_tokens=2048, temperature=0.85, top_p=1, do_sample=True, dump_result=""):
        input_data_ = []
        max_len = 0
        for job in range(self.dynamic_batch_size):
            for i in range(self.static_batch_size):
                input_data_.append(prompt)
                input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
                max_len = max(max_len, len(input_ids))
        
        base_round = max_len // self.base_length
        m = max_len % self.base_length

        if m != 0:
            base_round += 1
        
        total_pre = base_round * self.base_length
        inputs = []

        questions = []
        base_dynamic_batch = 0
        for job in range(self.dynamic_batch_size):
            one_batch = []
            one_pos = []
            batch_input = []
            tokens = self.tokenizer(
                input_data_[
                    base_dynamic_batch : base_dynamic_batch + self.static_batch_size
                ],
                padding="max_length",
                max_length=total_pre,
            )

            input_ids = np.array(tokens["input_ids"]).astype("int32")
            batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
            batch_input.append(input_ids)

            token_mask = torch.tensor(tokens["attention_mask"])
            pos = np.ones(input_ids.shape)
            pos_sum = [i.sum().item() for i in token_mask]
            for idx, pp in enumerate(pos_sum):
                pos[idx, -pp:] = np.arange(pp)
            pos = pos.astype("int32")
            batch_input.append(pos)

            mask = deepcopy(token_mask)
            mask_cond = torch.arange(mask.size(-1))
            mask = 1 - (mask_cond < (mask_cond + 1).view(mask.size(-1), 1)).to(torch.int32)
            mask = mask[None, None, :, :].expand(self.static_batch_size, 1, input_ids_seq_length, input_ids_seq_length)
            inc_mask = 1.0 - token_mask
            inc_mask = inc_mask[:, None, None, :].expand(self.static_batch_size, 1, input_ids_seq_length, input_ids_seq_length)
            base_mask = (inc_mask + mask).numpy()
            attention_mask = np.ones((self.static_batch_size, 1, input_ids_seq_length, total))
            attention_mask[..., :input_ids_seq_length] = base_mask
            attention_mask = attention_mask.astype("bool")

            batch_input.append(attention_mask)
            inputs.append(batch_input)
            base_dynamic_batch += self.static_batch_size
        
        outputs = self.model.inference_use_cache(
            inputs,
            max_new_tokens,
            tokens=tokens,
            base_round=base_round,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        
        res = []
        index = 0
        batch_count = 0
        for job in range(self.dynamic_batch_size):
            for s in range(self.static_batch_size):
                out_ids = outputs[batch_count][job][s]
                results = self.tokenizer.decode(out_ids)
                res.append(out_ids)
                answer = results
                index = index + 1
                if index >= len(input_data_):
                    break
        
        if dump_result != "":
            with open(dump_result, "w") as f:
                pk.dump(res, f)
        return answer, max_len


class InferModel:

    def __init__(self, config, ngf_path, model):
        self.input_names = []

        self.dev_ids = []
        self.dev_dram_limit = []
        self.dump_golden = 0
        self.split_strategy = 0
        self.engine_version = 0
        self.dump_result = ""

        self.static_batch_size = 1
        self.dynamic_batch_size = 1
        self.engine = "/home/FlagPerf/data/llama2_7b/llama2_7b_2048_bf16_q/llama2_7b_2048_bf16_q_multi.ngf"
        self.max_batch_size = 1
        self.input_index = 0
        self.base_length = 256
        self.max_new_tokens = 1
        self.temperature = 0.8
        self.top_p = 1.2
        self.do_sample = True
        global engine_version
        engine_version = 0
        global total
        total = 2048
        self.top_k = 0
        self.constant_output = 0
        self.typical_p = 0.5
        self.repeat_penalty = 0.
        self.presence_penalties = 0.
        self.frequency_penalties = 0.

        self.default_questions = ["hello"]
        self.input_data = self.default_questions
        self.tokenizer_path = "/home/FlagPerf/data/dataset/llama2_7b_hf"
        self.g = Graph(
            self.engine,
            self.tokenizer_path,
            self.static_batch_size,
            self.dynamic_batch_size,
            self.input_index,
            self.max_batch_size,
            self.base_length,
            constant_output=self.constant_output,
        )


    def __call__(self, model_inputs: list):
        
        prompt = model_inputs
        y = self.g.graph_run(
            self.input_data,
            prompt,
            self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            dump_result=self.dump_result,
        )

        tokens = y[1]
        res = y[0]

        foo_time = 0.0
        return res, foo_time, tokens

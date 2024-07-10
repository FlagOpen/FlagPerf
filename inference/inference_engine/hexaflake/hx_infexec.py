import os
import time
import torch
import argparse
import numpy as np
from ctypes import *
from tqdm import tqdm
from copy import deepcopy
from transformers.generation.utils import GenerationMixin
from transformers import (
    PretrainedConfig,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    LogitsProcessor,
)

from tcinfer import TcInfer


infer_lib = cdll.LoadLibrary("libtcinfer.so")


def align(size):
    if size % 128 == 0:
        return size
    else:
        return ((size // 128) + 1) * 128


def check_ret(ret, msg):
    if ret is None or ret != 0:
        print(msg)
        exit(1)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class GLMInfer(GenerationMixin):
    def __init__(self, engine_file, **kwargs):
        """A inference class
        Args:
        engine_file: engine file name
        the value of data_type includes ['int8','uint8','bfloat16','float32']
        batch_size: the batch size of inference
        max_batch_size: a config parameter for building engine
        """
        self.engine_file = engine_file
        self.batch_size = (
            1 if kwargs.get("batch_size") is None else kwargs["batch_size"]
        )
        self.static_batch_size = (
            1
            if kwargs.get("static_batch_size") is None
            else kwargs["static_batch_size"]
        )
        self.batch_count = (
            1 if kwargs.get("batch_count") is None else kwargs["batch_count"]
        )
        self.batch_sync = (
            1000 if kwargs.get("batch_sync") is None else kwargs["batch_sync"]
        )
        self.data_type = (
            "float32" if kwargs.get("data_type") is None else kwargs["data_type"]
        )
        self.max_batch_size = (
            0 if kwargs.get("max_batch_size") is None else kwargs["max_batch_size"]
        )
        self.print_throughput = (
            False
            if kwargs.get("print_throughput") is None
            else kwargs["print_throughput"]
        )
        self.in_out_nparray = (
            False if kwargs.get("in_out_nparray") is None else kwargs["in_out_nparray"]
        )
        self.input_files = "" if kwargs.get("inputs") is None else kwargs["inputs"]
        self.output_dir = (
            "" if kwargs.get("output_dir") is None else kwargs["output_dir"]
        )
        self.use_cache = (
            False if kwargs.get("use_cache") is None else kwargs["use_cache"]
        )
        self.rl2 = -1 if kwargs.get("resident_l2") is None else kwargs["resident_l2"]
        self.dev_ids_ = [] if kwargs.get("dev_ids") is None else kwargs["dev_ids"]
        self.dev_dram_limit = (
            [] if kwargs.get("dev_dram_limit") is None else kwargs["dev_dram_limit"]
        )
        self.dump_golden = (
            0 if kwargs.get("dump_golden") is None else kwargs["dump_golden"]
        )
        self.split_strategy = (
            0 if kwargs.get("split_strategy") is None else kwargs["split_strategy"]
        )
        self.config_file = (
            "" if kwargs.get("config_file") is None else kwargs["config_file"]
        )
        self.base_length = (
            256 if kwargs.get("base_length") is None else kwargs["base_length"]
        )
        self.engine_version = (
            0 if kwargs.get("engine_version") is None else kwargs["engine_version"]
        )
        self.total = 2048 if kwargs.get("total") is None else kwargs["total"]
        self.constant_output = (
            False
            if kwargs.get("constant_output") is None
            else kwargs["constant_output"]
        )

        self.__check_args()
        self.released_ = False
        if len(self.dev_ids_) == 0:
            self.dev_ids_ = [0]
        print("self.engine_file: ", self.engine_file)
        self.inf_obj = TcInfer(
            self.engine_file,
            use_cache=True,
            max_batch_size=self.max_batch_size,
            dev_ids=self.dev_ids_,
            dev_dram_limit=self.dev_dram_limit,
            dump_golden=self.dump_golden,
            split_strategy=self.split_strategy,
            engine_version=self.engine_version,
        )
        self.random_input = False
        self.next_input = []
        self.type_size_ = {1: "uint8", 2: "bfloat16", 4: "float32"}
        self.base_output_shift_ = 0

        self.base_infer_time = 0
        self.inc_infer_time = 0
        self.max_new_tokens = 32

        self.infered_tokens = 0

        self.pos_ = None
        self.attention_mask = None
        if self.static_batch_size > 1:
            self.multi_batch_finish = np.array([False] * self.static_batch_size)

        self.base_round = 1

        if not os.path.isfile(self.config_file):
            check_ret(None, "{} not exists.".format(self.config_file))
        self.config = PretrainedConfig().from_json_file(self.config_file)
        self.generation_config = GenerationConfig().from_model_config(self.config)
        self.output = []

    def __check_args(self):
        if self.engine_file == "":
            check_ret(1, "Missing engine file")
        if not os.path.exists(self.engine_file):
            check_ret(1, "engine file:{} not exitsts.".format(self.engine_file))
        if self.output_dir != "":
            if not os.path.exists(self.output_dir):
                msg = "output directory:{} not exists".format(self.output_dir)
                check_ret(1, msg)

    def __deal_use_cache_output(self, model_idx):
        finish_token = 2
        self.all_finished = True
        if model_idx == 0:
            self.use_cache_output_ = {}
            self.finish_flag_ = {}
        static_batches = len(self.next_input[0])
        for job in range(self.batch_size):
            if self.use_cache_output_.get(job) is None:
                self.use_cache_output_[job] = {}
                self.finish_flag_[job] = False

            if not self.finish_flag_[job]:
                for static_batch in range(static_batches):
                    if self.use_cache_output_[job].get(static_batch) is None:
                        self.use_cache_output_[job][static_batch] = []
                    if (
                        len(self.use_cache_output_[job][static_batch]) == 0
                        or self.use_cache_output_[job][static_batch][-1] != finish_token
                    ):
                        self.use_cache_output_[job][static_batch].append(
                            self.next_input[job][static_batch]
                        )

                next_token = torch.tensor(self.next_input[job])
                if static_batches > 1 and not self.constant_output:
                    mask_token = next_token == 2
                    self.multi_batch_finish[mask_token] = True
                    if all(self.multi_batch_finish):
                        self.all_finished = True
                        return
                if self.stop_inference(next_token, job):
                    self.finish_flag_[job] = True
                else:
                    self.all_finished = False

    def get_static_batch(self):
        return self.inf_obj.get_static_batch()

    def set_batch_size(self, batch_size):
        if self.batch_size == batch_size:
            return
        self.batch_size = batch_size
        self.inf_obj.set_batch_size(batch_size)

    def set_batch_count(self, batch_count):
        self.batch_count = batch_count

    def __internal_post_process(self, model_idx, token_idx=0):
        self.next_input.clear()
        output_binding_idx = 3
        for job in range(self.batch_size):
            out_array = self.inf_obj.get_output_by_binding_index(
                job, output_binding_idx, model_idx
            )
            if model_idx == 0 and self.base_output_shift_ == 0:
                self.base_output_shift_ = self.base_length
            one_batch = []
            next_token = self.get_next_token(out_array[:, -1, :], job)
            for i in range(self.static_batch_size):
                one_batch.append(int(next_token[i]))
            self.next_input.append(one_batch)

    def __infer_base(self, inputs):
        model_index = 0
        base = 0
        for r in range(self.base_round):
            for i in range(len(inputs)):
                input_ids, pos, attention_mask = inputs[i]
                input_ids_ = input_ids[:, base : base + self.base_length]
                pos_ = pos[:, base : base + self.base_length]
                partial_mask = np.concatenate(
                    (
                        np.zeros([1, 1, 1, base + self.base_length], dtype="bool"),
                        np.ones(
                            [1, 1, 1, self.total - base - self.base_length],
                            dtype="bool",
                        ),
                    ),
                    axis=-1,
                )
                attention_mask_ = (
                    attention_mask[:, :, base : base + self.base_length :, :]
                    | partial_mask
                )
                self.inf_obj.set_input(i, "pos", pos_)
                self.inf_obj.set_input(i, "input_ids", input_ids_)
                self.inf_obj.set_input(i, "attention_mask", attention_mask_)

            # batch_size, r, model_idx
            self.inf_obj.run(self.batch_size, r, model_index)
            base += self.base_length
        self.__internal_post_process(model_index)

    def __get_mask(self, job_index, token_idx):
        mask_ids = self.tokens["attention_mask"]
        new_mask = torch.cat(
            (torch.tensor(mask_ids), torch.ones((self.static_batch_size, token_idx + 1))), -1
        )
        expanded_mask = new_mask[:, None, None, :].expand(
            self.static_batch_size, 1, 1, new_mask.shape[-1]
        )
        inverted_mask = 1.0 - expanded_mask
        inc_mask = np.ones((self.static_batch_size, 1, 1, self.total))
        inc_mask[..., : new_mask.shape[-1]] = inverted_mask
        inc_mask = inc_mask.astype("bool")
        return inc_mask

    def __set_binding_inc(self, token_idx):
        model_idx = 1
        for job_index in range(self.batch_size):
            input_ids = np.array(self.next_input[job_index]).astype("int32")
            pos = np.array(self.pos_sum[job_index]).astype("int32")
            self.inf_obj.set_input(job_index, "pos", pos, model_idx)
            mask = self.__get_mask(job_index, token_idx)
            self.inf_obj.set_input(job_index, "input_ids", input_ids, model_idx)
            self.inf_obj.set_input(job_index, "attention_mask", mask, model_idx)

    def __infer_incremental(self, token_idx):
        model_idx = 1
        self.__set_binding_inc(token_idx)
        context = token_idx + self.base_output_shift_
        self.inf_obj.run(self.batch_size, context, model_idx)
        self.__internal_post_process(model_idx, token_idx)

    def show_throughput(self):
        base_spent_time = self.base_infer_time * 1000000
        self.static_batch_size = self.get_static_batch()
        jobs = self.batch_count * self.static_batch_size * self.batch_size
        total_tokens = jobs * self.infered_tokens
        base_tokens = jobs * self.base_round
        inc_spent_time = self.inc_infer_time * 1000000

        print("============================================")
        print("base infer time: {:.6f} (s)".format(self.base_infer_time))
        print(
            "base throughput: {:.3f} (tokens/s)".format(
                (base_tokens * 1000000) / base_spent_time
            )
        )
        inc_tokens = jobs * (self.infered_tokens - 1)
        print("incremental infer time: {:.6f} (s)".format(self.inc_infer_time))
        print(
            "incremental throughput: {:.3f} (tokens/s)".format(
                (inc_tokens * 1000000) / inc_spent_time
            )
        )

        print(
            "infer time: {:.6f} (s)".format(self.base_infer_time + self.inc_infer_time)
        )
        print(f"infer tokens: {total_tokens}")
        print(
            "average throughput: {:.3f} (tokens/s)".format(
                (total_tokens * 1000000) / (inc_spent_time + base_spent_time)
            )
        )
        print("============================================")

    def generate_pos_attention_mask(self, inputs):
        self.pos_ = []
        self.attention_mask = []
        self.pos_sum = []
        for inp in inputs:
            self.pos_.append(inp[1].astype("int32"))
            self.attention_mask.append(inp[2])
            self.pos_sum.append(inp[1][:, -1] + 1)

    def set_temperature(self, inputs, temperature=0.85, top_p=1, do_sample=False):
        self.input_ids = []
        for dynamic_batch_idx in range(self.batch_size):
            self.input_ids.append(inputs[dynamic_batch_idx][0])

        self.generation_config.temperature = temperature
        self.generation_config.top_p = top_p
        self.generation_config.do_sample = do_sample
        self.generation_config.top_k = self.top_k
        self.generation_config.repetition_penalty = self.repeat_penalty
        self.generation_config.typical_p = self.typical_p

        batch_size, input_ids_seq_length = (
            self.input_ids[0].shape[0],
            self.input_ids[0].shape[-1],
        )

        generation_config = self.generation_config
        bos_token_id, eos_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        self.eos_token_id_tensor = torch.tensor(eos_token_id)

        has_default_max_length = generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            pass
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_seq_length
            )

        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())

        self.logits_processor = []
        for dynamic_batch_idx in range(self.batch_size):
            self.logits_processor.append(
                self._get_logits_processor(
                    generation_config=generation_config,
                    input_ids_seq_length=input_ids_seq_length,
                    encoder_input_ids=self.input_ids[dynamic_batch_idx].astype("int64"),
                    prefix_allowed_tokens_fn=None,
                    logits_processor=logits_processor,
                )
            )
        self.stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=StoppingCriteriaList(),
        )
        self.logits_warper = self._get_logits_warper(generation_config)

        self.scores = None

    def _apply_penalties(self, logits):
        """
        reference: vllm
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        """
        output_tokens_tensor = torch.tensor(self.output)
        vocab_size = self.config.vocab_size
        bin_counts = torch.zeros((self.static_batch_size, vocab_size + 1), dtype=torch.long)
        bin_counts.scatter_add_(1, output_tokens_tensor, torch.ones_like(output_tokens_tensor))
        bin_counts = bin_counts[:, :vocab_size]
        frequency_penalties = torch.tensor([[self.frequency_penalties]], dtype=logits.dtype)
        presence_penalties = torch.tensor([[self.presence_penalties]], dtype=logits.dtype)
        logits = logits - frequency_penalties * bin_counts
        logits = logits - presence_penalties * (bin_counts > 0)
        return logits

    def _do_sample(self, next_token_scores):
        if self.generation_config.do_sample:
            probs = torch.softmax(next_token_scores, dim=-1).numpy()
            next_token = (
                torch.multinomial(torch.Tensor(probs), num_samples=1).squeeze(1).numpy()
            )
        else:
            next_token = torch.argmax(next_token_scores, dim=-1).numpy()
        return next_token

    def get_next_token(self, next_token_logits, dynamic_batch_idx):
        # pre-process distribution
        t_input_ids = torch.LongTensor(self.input_ids[dynamic_batch_idx])
        next_token_scores = torch.Tensor(next_token_logits)
        next_token_scores = self.logits_processor[dynamic_batch_idx](
            t_input_ids, next_token_scores
        )
        next_token = self._do_sample(next_token_scores)
        if len(self.output) > 0 and (self.frequency_penalties != 0. or self.presence_penalties != 0.):
            for _b, _token in enumerate(next_token.flatten().tolist()):
                self.output[_b].append(_token)
            next_token_logits = self._apply_penalties(next_token_scores)
            next_token = self._do_sample(next_token_logits)

        return next_token

    def stop_inference(self, next_token, dynamic_batch_idx):
        t_input_ids = torch.LongTensor(self.input_ids[dynamic_batch_idx])
        unfinished_sequences = t_input_ids.new(t_input_ids.size(0)).fill_(1)
        self.input_ids[dynamic_batch_idx] = np.concatenate(
            [self.input_ids[dynamic_batch_idx], next_token[:, None]], axis=-1
        )
        unfinished_sequences = unfinished_sequences.mul(
            next_token.tile(self.eos_token_id_tensor.shape[0], 1)
            .ne(self.eos_token_id_tensor.unsqueeze(1))
            .prod(dim=0)
        )
        # stop when each sentence is finished, or if we exceed the maximum length
        if not self.constant_output:
            if unfinished_sequences.max() == 0 or self.stopping_criteria(
                self.input_ids[dynamic_batch_idx], self.scores
            ):
                return True
        return False

    def inference_use_cache(
        self,
        inputs: list,
        max_new_tokens,
        tokens=None,
        base_round=1,
        temperature=0.85,
        top_p=1,
        top_k=0.,
        do_sample=False,
        typical_p = 0.1,
        repeat_penalty= 1.0,
        presence_penalties = 0.,
        frequency_penalties = 0.,
    ):
        if not self.use_cache:
            print("engine not created with user cache.")
            return
        repeat_penalty = 1.0
        self.typical_p = typical_p
        self.repeat_penalty = repeat_penalty
        self.top_k = top_k
        self.presence_penalties = presence_penalties
        self.frequency_penalties = frequency_penalties
        self.set_temperature(
            inputs, temperature=temperature, top_p=top_p, do_sample=do_sample
        )

        output = []
        self.tokens = tokens
        self.max_new_tokens = max_new_tokens
        total_steps = self.batch_count * self.max_new_tokens
        self.base_infer_time = 0
        self.inc_infer_time = 0
        self.base_round = base_round
        self.generate_pos_attention_mask(inputs)
        self.inf_obj.clear_use_cache()
        pbar = tqdm(total=total_steps, desc="Infering")
        for i in range(self.batch_count):
            self.base_output_shift_ = 0
            self.infered_tokens = 0
            t0 = time.perf_counter()
            self.__infer_base(inputs)
            self.__deal_use_cache_output(0)
            t1 = time.perf_counter()
            self.base_infer_time = self.base_infer_time + (t1 - t0)
            total_steps = total_steps - 1
            self.infered_tokens = self.infered_tokens + self.base_round
            self.base_output_shift_ += (self.base_round - 1) * self.base_length
            pbar.update(1)
            self.output = deepcopy([self.use_cache_output_[0][i] for i in self.use_cache_output_[0]])
            for token_idx in range(max_new_tokens - 1):
                self.__infer_incremental(token_idx)
                self.__deal_use_cache_output(1)
                pbar.update(1)
                self.infered_tokens = self.infered_tokens + 1
                total_steps = total_steps - 1
                for job, _ in enumerate(self.pos_sum):
                    if self.static_batch_size > 1 and not self.constant_output:
                        pos_mask = (1 - self.multi_batch_finish).astype("bool")
                        self.pos_sum[job][pos_mask] += 1
                    else:
                        self.pos_sum[job] += 1
                if self.all_finished:
                    pbar.update(total_steps)
                    break
            t2 = time.perf_counter()
            self.inc_infer_time = self.inc_infer_time + (t2 - t1)
            output.append(self.use_cache_output_)
        return output


def main():
    parser = argparse.ArgumentParser(
        description="This program is used to" " infer with python api"
    )
    parser.add_argument(
        "--engine-file",
        action="store",
        type=str,
        default="",
        help="set the engine file",
    )
    parser.add_argument(
        "--batch-size",
        action="store",
        type=int,
        default=1,
        help="default 1, set the size of every batch",
    )
    parser.add_argument(
        "--batch-count",
        action="store",
        type=int,
        default=1,
        help="default 1, set the batch count of infer",
    )
    parser.add_argument(
        "--batch_sync",
        action="store",
        type=int,
        default=1000,
        help="default 1000, set number of batchs do a synchronize",
    )
    parser.add_argument(
        "--max-batch-size",
        action="store",
        type=int,
        default=0,
        help="default 0, set max batch size",
    )
    parser.add_argument(
        "--inputs",
        action="store",
        type=str,
        default="",
        help="default with random inputs, format:"
        "job1_tensor1,job1_tensor2,...:job2_tensor1,job2_tensor2,...",
    )
    parser.add_argument(
        "--output-dir",
        action="store",
        type=str,
        default="",
        help="set the dump directory of output tensors, "
        "output tensor subpath format is "
        "job_{job No}/{tensor name}.dat",
    )

    args = parser.parse_args()

    TcInfer(
        args.engine_file,
        batch_size=args.batch_size,
        batch_count=args.batch_count,
        batch_sync=args.batch_sync,
        max_batch_size=args.max_batch_size,
        inputs=args.inputs,
        output_dir=args.output_dir,
        print_throughput=True,
        in_out_nparray=False,
    )


if __name__ == "__main__":
    main()


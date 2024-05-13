import os
import subprocess
import time
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from utils import load_tokenizer, read_model_name
import torch

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
        tllm_checkpoint = os.path.join(config.data_dir, "tllm_checkpoint_" + num_gpus + "gpu" + "_tp" + tp_size + "_pp" + pp_size)

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

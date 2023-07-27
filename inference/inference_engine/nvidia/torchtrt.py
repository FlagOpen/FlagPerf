import os
import torch
import torch_tensorrt as torchtrt
import time


class InferModel:

    def __init__(self, config, onnx_path, model):
        self.config = config
        self.origin_model = model
        self.traced_model = None
        self.trt_model = None
        self.full_compile = config.torchtrt_full_compile

    def __call__(self, model_inputs: list):
        start = time.time()
        
        if self.traced_model is None:            
            self.traced_model = torch.jit.trace(self.origin_model,
                                                model_inputs)
            self.trt_model = torchtrt.compile(
                self.traced_model,
                inputs=model_inputs,
                truncate_long_and_double=True,
                enabled_precisions={torch.float32, torch.float16},
                require_full_compilation=self.full_compile)

        compile_foo_time = time.time() - start
        
        model_outputs = self.trt_model(*model_inputs)
        return [model_outputs], compile_foo_time

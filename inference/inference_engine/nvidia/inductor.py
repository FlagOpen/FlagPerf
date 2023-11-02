import os
import torch
import torch._dynamo

import time


class InferModel:

    def __init__(self, config, onnx_path, model):
        self.config = config
        torch._dynamo.reset()
        self.model = torch.compile(model, mode=config.dynamo_mode, dynamic=config.dynamo_dynamic)
        self.warmup = config.dynamo_wamrup_iters

    def __call__(self, model_inputs: list):
        start = time.time()
        if self.warmup != 0:
            for i in range(self.config.dynamo_wamrup_times):
                _ = self.model(model_inputs[0])
            self.warmup -= 1

        torch.cuda.synchronize()
        compile_foo_time = time.time() - start

        model_outputs = self.model(model_inputs[0].cuda())
        return [model_outputs], compile_foo_time

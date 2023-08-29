import onnx
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.relay import param_dict
from tvm.contrib import xpu_config
import torch
import os
import subprocess
from loguru import logger
import numpy as np
import time

class InferModel:
    
    def __init__(self, config , onnx_path, model):
        self.input_names = []
        self.engine = self.build_engine(config, onnx_path)

    def build_engine(self, config, onnx_path):
        onnx_model = onnx.load(onnx_path)
        shape_dict = {}
        for input in onnx_model.graph.input:
            input_shape = input.type.tensor_type.shape.dim
            input_shape = [a.dim_value for a in input_shape]
            input_shape[0] = config.batch_size
            input_name = input.name #'inputs:0'
            self.input_names.append(input_name)
            shape_dict[input_name] = input_shape
        
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        target_host = f'llvm -acc=xpu{os.environ.get("XPUSIM_DEVICE_MODEL", "KUNLUN1")[-1]}'
        ctx = tvm.device("xpu", 0)
        build_config = {}
        if config.fp16 == True:
            os.environ["XTCL_USE_NEW_ALTER_PASS"] = '1'
            input_fp16 = { name:"float16" for name in self.input_names}
            build_config["XPUOutDtypeConfig"] = xpu_config.XPUOutDtypeConfig(
                                                 default_precision="float16",
                                                 config_last_node=True,
                                                 config_map={
                                                 },
                                                 config_var_dtype_map=input_fp16,
                                                 ).value()
        else: ## fp32
            os.environ['XTCL_USE_FP16'] = '0'
            os.environ['XTCL_QUANTIZE_WEIGHT'] = '0'

        with tvm.transform.PassContext(opt_level=3, config=build_config):
            vm_exec = relay.backend.vm.compile(mod,
                                             target=target_host,
                                             target_host=target_host,
                                             params=params)
        from tvm.runtime.vm import VirtualMachine
        vm = VirtualMachine(vm_exec, ctx)
        return vm

    def __call__(self, model_inputs: list):
        for index, input_name in enumerate(self.input_names):
            self.engine.set_one_input("main",input_name, tvm.nd.array(model_inputs[index]))
        self.engine.run()
        output_list = [self.engine.get_output(i) for i in range(self.engine.get_num_outputs())]
        foo_time_start = time.time()
        # d2h
        output_list = [torch.from_numpy(output.asnumpy()) for output in output_list]
        foo_time = time.time() - foo_time_start
        return output_list, foo_time



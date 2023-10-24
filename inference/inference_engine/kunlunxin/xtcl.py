import os
import time

import onnx
import torch
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor, xpu_config
from tvm.relay.xpu.patterns import custom_fuse_patterns
from tvm.runtime.vm import VirtualMachine


class InferModel:

    def __init__(self, config, onnx_path, model):
        self.input_names = []
        self.engine = self.build_engine(config, onnx_path)
        self.vm_enable = True

    def build_engine(self, config, onnx_path):
        onnx_model = onnx.load(onnx_path)
        shape_dict = {}
        for inp in onnx_model.graph.input:
            input_name, input_shape, _, _ = relay.frontend.onnx.get_info(inp)
            input_shape[0] = config.batch_size
            self.input_names.append(input_name)
            shape_dict[input_name] = input_shape

        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        target_host = f'llvm -acc=xpu{os.environ.get("XPUSIM_DEVICE_MODEL", "KUNLUN1")[-1]}'
        ctx = tvm.device("xpu", 0)
        build_config = config.build_config if 'build_config' in config._fields else {}
        disabled_pass = config.disabled_pass if 'disabled_pass' in config._fields else []
        self.vm_enable = config.vm_enable if 'vm_enable' in config._fields else True
        if "pattern_match" in build_config:
            build_config["XPUFuzzyMatch"] = xpu_config.XPUGraphMatchConfig(
                pattern_match=build_config["pattern_match"]).value()
            del build_config["pattern_match"]
        #os.environ["XTCL_BUILD_DEBUG"] = '1'
        if config.resnet50_fuse:
            os.environ["XTCL_FUSE_RES50V15"] = '1'
        if config.fp16 == True:
            os.environ["XTCL_USE_NEW_ALTER_PASS"] = '1'
            build_config["XPUOutDtypeConfig"] = xpu_config.XPUOutDtypeConfig(
                default_precision="float16",
                config_last_node=True,
                config_map={},
            ).value()
        else: ## fp32
            os.environ["XTCL_USE_NEW_ALTER_PASS"] = '1'
            os.environ['XTCL_USE_FP16'] = '1'
            os.environ['XTCL_QUANTIZE_WEIGHT'] = '1'

        with tvm.transform.PassContext(opt_level=3, config=build_config, disabled_pass=disabled_pass):
            if self.vm_enable:
                vm_exec = relay.backend.vm.compile(mod, target=target_host, target_host=target_host, params=params)
                vm = VirtualMachine(vm_exec, ctx)
                return vm
            else:
                graph, lib, params = relay.build(mod,
                                                 target="xpu -libs=xdnn -split-device-funcs -device-type=xpu2",
                                                 params=params)
                m = graph_executor.create(graph, lib, ctx)
                m.set_input(**params)
                return m

    def __call__(self, model_inputs: list):
        for index, input_name in enumerate(self.input_names):
            if self.vm_enable:
                self.engine.set_one_input("main", input_name, model_inputs[index].numpy())
            else:
                self.engine.set_input(input_name, tvm.nd.array(model_inputs[index]))
        self.engine.run()
        foo_time_start = time.time()
        output_list = [self.engine.get_output(i) for i in range(self.engine.get_num_outputs())]
        # d2h
        output_list = [torch.from_numpy(output.numpy()) for output in output_list]
        foo_time = time.time() - foo_time_start
        return output_list, foo_time

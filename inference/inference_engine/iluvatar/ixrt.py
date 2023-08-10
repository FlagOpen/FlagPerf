from ixrt import IxRT, RuntimeConfig, RuntimeContext
import torch
import os
import subprocess
from loguru import logger
import numpy as np
import time


class InferModel:

    class HostDeviceMem(object):

        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def __init__(self, config, onnx_path, model):
        self.str_to_numpy_dict = {
            "int32": np.int32,
            "float16": np.float16,
            "float32": np.float32,
        }
        self.engine = self.build_engine(config, onnx_path)
        self.outputs = self.allocate_buffers(self.engine)

    def config_init_engine(self, config, onnx_path):
        quant_file = None

        runtime_config = RuntimeConfig()

        input_shapes = [config.batch_size, 3, config.image_size, config.image_size]
        runtime_config.input_shapes = [("input", input_shapes)]
        runtime_config.device_idx = 0

        precision = "float16"
        if precision == "int8":
            assert quant_file, "Quant file must provided for int8 inferencing."

        runtime_config.runtime_context = RuntimeContext(
            precision,
            "nhwc",
            use_gpu=True,
            pipeline_sync=True,
            input_types=config.input_types,
            output_types=config.output_types,
            input_device="gpu",
            output_device="gpu",
        )

        runtime = IxRT.from_onnx(onnx_path, quant_file, runtime_config)
        return runtime

    def build_engine(self, config, onnx_path):
        if config.exist_compiler_path is None:
            output_path = config.log_dir + "/" + config.ixrt_tmp_path

            dir_output_path = os.path.dirname(output_path)
            os.makedirs(dir_output_path, exist_ok=True)

            time.sleep(10)

            runtime = self.config_init_engine(config, onnx_path)
            print(f"Build Engine File: {output_path}")
            runtime.BuildEngine()
            runtime.SerializeEngine(output_path)
            print("Build Engine done!")
        else:
            output_path = config.exist_compiler_path
            print(f"Use existing engine: {output_path}")

        runtime = IxRT()
        runtime.LoadEngine(output_path, config.batch_size)
        return runtime

    def allocate_buffers(self, engine):
        output_map = engine.GetOutputShape()
        output_io_buffers = []
        output_types = {}
        config = engine.GetConfig()
        for key, val in config.runtime_context.output_types.items():
            output_types[key] = str(val)
        for name, shape in output_map.items():
            # 1. apply memory buffer for output of the shape
            buffer = np.zeros(
                shape.dims, dtype=self.str_to_numpy_dict[output_types[name]]
            )
            buffer = torch.tensor(buffer).cuda()
            # 2. put the buffer to a list
            output_io_buffers.append([name, buffer, shape])

        engine.BindIOBuffers(output_io_buffers)
        return output_io_buffers

    def __call__(self, model_inputs: list):
        batch_size = np.unique(np.array([i.size(dim=0) for i in model_inputs]))
        batch_size = batch_size[0]
        input_map = self.engine.GetInputShape()
        input_io_buffers = []

        for i, model_input in enumerate(model_inputs):
            model_input = torch.tensor(model_input.numpy(), dtype=torch.float32).cuda()
            if not model_input.is_contiguous():
                model_input = model_input.contiguous()
            name, shape = list(input_map.items())[0]
            _shape, _padding = shape.dims, shape.padding
            _shape = [i + j for i, j in zip(_shape, _padding)]
            _shape = [_shape[0], *_shape[2:], _shape[1]]
            input_io_buffers.append([name, model_input, shape])

        self.engine.BindIOBuffers(self.outputs)
        self.engine.LoadInput(input_io_buffers)

        # torch.cuda.synchronize()
        self.engine.Execute()
        # torch.cuda.synchronize()

        gpu_io_buffers = []
        for buffer in self.outputs:
            # gpu_io_buffers.append([buffer[0], buffer[1], buffer[2]])
            gpu_io_buffers.append(buffer[1])

        return gpu_io_buffers, 0

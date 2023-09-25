import os
import torch
from torch import autocast
import tensorrt as trt

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import subprocess


class InferModel:

    class HostDeviceMem(object):

        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(
                self.device)

        def __repr__(self):
            return self.__str__()

    def __init__(self, config, onnx_path, model):
        self.config = config

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        self.engine = self.build_engine(config, onnx_path)

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(
            self.engine)

        self.context = self.engine.create_execution_context()
        self.numpy_to_torch_dtype_dict = {
            bool: torch.bool,
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
        }
        self.str_to_torch_dtype_dict = {
            "bool": torch.bool,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }

    def build_engine(self, config, onnx_path):
        if config.exist_compiler_path is None:
            trt_path = config.log_dir + "/" + config.ixrt_tmp_path

            dir_trt_path = os.path.dirname(trt_path)
            os.makedirs(dir_trt_path, exist_ok=True)

            time.sleep(10)

            trtexec_cmd = "ixrtexec --onnx=" + onnx_path + " --save_engine=" + trt_path
            if config.fp16:
                trtexec_cmd += " --precision fp16"
            if config.has_dynamic_axis:
                trtexec_cmd += " --minShapes=" + config.minShapes
                trtexec_cmd += " --optShapes=" + config.optShapes
                trtexec_cmd += " --maxShapes=" + config.maxShapes

            p = subprocess.Popen(trtexec_cmd, shell=True)
            p.wait()
        else:
            trt_path = config.exist_compiler_path

        with open(trt_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in range(engine.num_bindings):
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, model_inputs: list):

        for i, model_input in enumerate(model_inputs):
            model_input = model_input.cuda()

            cuda.memcpy_dtod_async(
                self.inputs[i].device,
                model_input.data_ptr(),
                model_input.element_size() * model_input.nelement(),
                self.stream,
            )

        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        result = []
        for out in self.outputs:
            out_tensor = torch.empty(out.host.shape, device="cuda").to(
                self.str_to_torch_dtype_dict[str(out.host.dtype)])
            cuda.memcpy_dtod_async(
                out_tensor.data_ptr(),
                out.device,
                out_tensor.element_size() * out_tensor.nelement(),
                self.stream,
            )
            result.append(out_tensor)

        self.stream.synchronize()
        return result, 0

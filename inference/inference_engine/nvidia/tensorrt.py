import os
import torch
from torch import autocast
import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")
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

    def build_engine(self, config, onnx_path):
        if config.exist_compiler_path is None:
            trt_path = config.log_dir + "/" + config.trt_tmp_path

            dir_trt_path = os.path.dirname(trt_path)
            os.makedirs(dir_trt_path, exist_ok=True)

            time.sleep(10)

            trtexec_cmd = "trtexec --onnx=" + onnx_path + " --saveEngine=" + trt_path
            if config.fp16:
                trtexec_cmd += " --fp16"
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

        for binding in engine:
            size = trt.volume(
                engine.get_binding_shape(binding)) * engine.max_batch_size
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

        batch_size = np.unique(np.array([i.size(dim=0) for i in model_inputs]))
        batch_size = batch_size[0]

        for i, model_input in enumerate(model_inputs):
            binding_name = self.engine[i]
            binding_dtype = trt.nptype(
                self.engine.get_binding_dtype(binding_name))
            model_input = model_input.to(
                self.numpy_to_torch_dtype_dict[binding_dtype])

            cuda.memcpy_dtod_async(
                self.inputs[i].device,
                model_input.data_ptr(),
                model_input.element_size() * model_input.nelement(),
                self.stream,
            )

        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()

        return [
            torch.from_numpy(out.host.reshape(batch_size, -1))
            for out in self.outputs
        ], 0

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import os
import subprocess
from loguru import logger
import numpy as np


def build_engine(config):
    onnx_path = config.perf_dir + "/" + config.onnx_path
    trt_path = config.log_dir + "/" + config.trt_tmp_path

    dir_trt_path = os.path.dirname(trt_path)
    os.makedirs(dir_trt_path, exist_ok=True)

    trtexec_cmd = "trtexec --onnx=" + onnx_path + " --saveEngine=" + trt_path
    if config.fp16:
        trtexec_cmd += " --fp16"
    if config.has_dynamic_axis:
        trtexec_cmd += " --minShapes=" + config.minShapes
        trtexec_cmd += " --optShapes=" + config.optShapes
        trtexec_cmd += " --maxShapes=" + config.maxShapes

    subprocess.run(trtexec_cmd).wait()

    trtlogger = trt.Logger()

    logger.debug("Using exist " + trt_path)
    with open(trt_path, "rb") as f, trt.Runtime(trtlogger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def get_inference_toolkits(config):
    engine = build_engine(config)
    return (engine, allocate_buffers, inference, postprocess_the_outputs)


class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        size = abs(size)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

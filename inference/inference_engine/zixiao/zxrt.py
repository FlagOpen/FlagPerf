import onnx
import onnxruntime
import torch
import os
import subprocess
from loguru import logger
import numpy as np
import time
import TopsInference

def type2dtype(types):
    dtypes = []
    for elem_type in types:
        if elem_type == 1:
            dtypes.append(TopsInference.DT_FLOAT32)
        elif elem_type == 7:
            dtypes.append(TopsInference.DT_INT64)
        elif elem_type == 6:
            dtypes.append(TopsInference.DT_INT32)
        elif elem_type == 3:
            dtypes.append(TopsInference.DT_INT8)
        elif elem_type == 4:
            dtypes.append(TopsInference.DT_UINT8)
        elif elem_type == 9:
            dtypes.append(TopsInference.DT_BOOL)
        elif elem_type == 10:
            dtypes.append(TopsInference.DT_FLOAT16)
        else:
            raise Exception("unknown default dtypes:{}, {}".format(elem_type))
    return dtypes

class InferModel:

    def __init__(self, config, onnx_path, model):
        self.input_names = []
        self.engine = self.build_engine(config, onnx_path)
        self.test_index = 0
        self.batch_size = config.batch_size
        self.zixiao_VG_num = 6

    def build_engine(self, config, onnx_path):
        self.handler = TopsInference.set_device(0, -1)
        onnx_model = onnx.load(onnx_path)
        self.input_shapes = []
        self.input_dtype = []
        for input in onnx_model.graph.input:
            input_shape = input.type.tensor_type.shape.dim
            input_shape = [a.dim_value for a in input_shape]
            input_shape[0] = config.batch_size // 6
            input_name = input.name
            self.input_names.append(input_name)
            self.input_shapes.append(input_shape)
            self.input_dtype.append(input.type.tensor_type.elem_type)
        self.input_dtype = type2dtype(self.input_dtype)
        if config.fp16 == True:
            set_input_dtype = []
            for tops_dtype in self.input_dtype:
                if tops_dtype == TopsInference.DT_FLOAT32:
                    set_input_dtype.append(TopsInference.DT_FLOAT16)
                else:
                    set_input_dtype.append(tops_dtype)
            self.input_dtype = set_input_dtype

        onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
        onnx_parser.set_input_names(self.input_names)
        onnx_parser.set_input_dtypes(self.input_dtype)
        onnx_parser.set_input_shapes(input_shape)

        network = onnx_parser.read(onnx_path)
        optimizer = TopsInference.create_optimizer()
        if config.fp16 == True: 
            optimizer.set_build_flag(TopsInference.KFP16_MIX)
        engine = optimizer.build(network)
        engine.save_executable(onnx_path+".bin")
        engine = TopsInference.load(onnx_path+".bin")
        self.streams = []
        for i in range(12):
            self.streams.append(TopsInference.create_stream())
        return engine

    def __call__(self, model_inputs: list):
        inputs = []
        outputs = []
        foo_time_start = time.time()
        for input in model_inputs:
            inputs.append(input.numpy())
        input_batch = inputs[0].shape[0]
        # zixiao acceleration card has 6 compute cells
        assert input_batch % self.zixiao_VG_num == 0
        vg_batch = input_batch // self.zixiao_VG_num
        foo_time = time.time() - foo_time_start
        for i in range(self.zixiao_VG_num):
            vg_input = []
            foo_time_start_data_slice = time.time()
            for input in inputs:
                vg_input.append(input[vg_batch * i: vg_batch * (i + 1)])
            foo_time += time.time() - foo_time_start_data_slice
            outputs.append(self.engine.runV2(vg_input,
                                             py_stream=self.streams[self.test_index % 12]))
            self.test_index += 1
        zx_outputs = []
        for i in range(self.zixiao_VG_num):
            zx_outputs.append([output for output in outputs[i].get()])
        # concat vg_batch result
        foo_time_start2 = time.time()
        host_output = []
        for i in range(len(zx_outputs[0])):
            tmp_output = []
            for j in range(self.zixiao_VG_num):
                tmp_output.append(zx_outputs[j][i])
            host_output.append(np.concatenate(tmp_output))
        infer_output = [torch.from_numpy(output) for output in host_output]
        foo_time += time.time() - foo_time_start2
        return infer_output, foo_time

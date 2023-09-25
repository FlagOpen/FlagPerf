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
        self.batch_size = config.zixiao_test_batch_size
        self.zixiao_VG_num = 6

    def build_engine(self, config, onnx_path):
        self.handler = TopsInference.set_device(4, -1)
        onnx_model = onnx.load(onnx_path)
        self.input_shapes = []
        self.input_dtype = []
        for input in onnx_model.graph.input:
            input_shape = input.type.tensor_type.shape.dim
            input_shape = [a.dim_value for a in input_shape]
            input_shape[0] = config.zixiao_test_batch_size
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
        total_input_num = inputs[0].shape[0]
        total_test_batch = (total_input_num + self.batch_size - 1) //  self.batch_size
        # zixiao acceleration card has 6 compute cells
        foo_time = time.time() - foo_time_start
        for i in range(total_test_batch):
            foo_time_start_data_slice = time.time()
            vg_input = []
            for input in inputs:
                vg_input.append(input[self.batch_size * i: self.batch_size * (i + 1)])
            foo_time += time.time() - foo_time_start_data_slice
            outputs.append(self.engine.runV2(vg_input, py_stream=self.streams[i % 12]))
        # zixiao sync
        for i in range(12):
            outputs[i-12].get()
        # zixiao sync done

        # concat batch result
        foo_time_start_d2h = time.time()
        zx_outputs = []
        for i in range(total_test_batch):
            zx_outputs.append([output for output in outputs[i].get()])
        host_output = []
        for i in range(len(zx_outputs[0])):
            tmp_output = []
            for j in range(total_test_batch):
                tmp_output.append(zx_outputs[j][i])
            host_output.append(np.concatenate(tmp_output))
        infer_output = [torch.from_numpy(output) for output in host_output]
        foo_time += time.time() - foo_time_start_d2h
        return infer_output, foo_time

# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib
from loguru import logger
import time
import os
import sys
from tools import init_logger, merge_config
from argparse import ArgumentParser


def main(config):

    init_logger(config)
    config = merge_config(config)
    # e.g. import funcs from benchmarks/resnet50/pytorch/__init__.py
    benchmark_module = importlib.import_module(
        "benchmarks." + config.case + "." + config.framework, __package__)
    """
    Init
    """
    logger.log("Init Begin", "building dataloader and model")
    start = time.time()

    dataloader = benchmark_module.build_dataloader(config)
    model = benchmark_module.create_model(config)

    duration = time.time() - start
    logger.log("Init End", str(duration) + " seconds")
    """
    Using framework.eval(like torch.eval) to validate model & dataloader
    """
    logger.log("Model Forward Begin", "")
    start = time.time()

    evaluator = benchmark_module.evaluator

    p_forward, p_forward_core, val_acc = benchmark_module.model_forward(
        model, dataloader, evaluator, config)

    logger.log("Model Forward End", "")
    if config.compiler is None:
        return config, p_forward, None, p_forward_core, None, val_acc, None
    """
    Convert model into onnx
    """
    logger.log("Export Begin",
               "Export " + config.framework + " model into .onnx")
    start = time.time()

    onnx_path = benchmark_module.export_model(model, config)

    duration = time.time() - start
    logger.log("Export End", str(duration) + " seconds")
    # e.g. import funcs from inference_engine/nvidia/inference.py
    vendor_module = importlib.import_module("inference_engine." +
                                            config.vendor + "." +
                                            config.compiler)
    """
    Compiling backend(like tensorRT)
    """
    logger.log("Vendor Compile Begin",
               "Compiling With " + config.vendor + "." + config.compiler)
    start = time.time()

    compile_model = vendor_module.InferModel(config, onnx_path, model)

    duration = time.time() - start
    logger.log("Vendor Compile End", str(duration) + " seconds")
    """
    inference using engine
    """
    logger.log("Vendor Inference Begin", "")
    start = time.time()

    p_infer, p_infer_core, infer_acc = benchmark_module.engine_forward(
        compile_model, dataloader, evaluator, config)

    logger.log("Vendor Inference End", "")

    return config, p_forward, p_infer, p_forward_core, p_infer_core, val_acc, infer_acc


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--perf_dir",
                        type=str,
                        required=True,
                        help="abs dir of FlagPerf/inference/")

    parser.add_argument("--data_dir",
                        type=str,
                        required=True,
                        help="abs dir of data used in dataloader")

    parser.add_argument("--log_dir",
                        type=str,
                        required=True,
                        help="abs dir to write log")

    parser.add_argument("--loglevel",
                        type=str,
                        required=True,
                        help="DEBUG/INFO/WARNING/ERROR")

    parser.add_argument("--case",
                        type=str,
                        required=True,
                        help="case name like resnet50")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    parser.add_argument("--framework",
                        type=str,
                        required=True,
                        help="validation framework name like pytorch")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


if __name__ == "__main__":
    config_from_args = parse_args()
    config_from_args.framework = config_from_args.framework.split('_')[0]

    e2e_start = time.time()

    config, p_forward, p_infer, p_forward_core, p_infer_core, val_acc, infer_acc = main(
        config_from_args)

    e2e_time = time.time() - e2e_start
    e2e_time = round(float(e2e_time), 3)

    flops = eval(config.flops) * (p_infer_core if p_infer_core is not None else p_forward_core)

    infer_info = {
        "vendor": config.vendor,
        "compiler": config.compiler,
        "precision": "fp16" if config.fp16 else "fp32",
        "batchsize": config.batch_size,
        "flops": flops,
        "e2e_time(second)": e2e_time,
        "p_validation_whole(qps)": p_forward,
        "p_validation_core(qps)": p_forward_core,
        "p_inference_whole(qps)": p_infer,
        "*p_inference_core(qps)": p_infer_core,
        "val_average_acc": val_acc,
        "infer_average_acc": infer_acc
    }
    logger.log("Finish Info", infer_info)

import importlib
from loguru import logger
import time
import os
import sys
from tools.drivers import init_logger, merge_config
from argparse import ArgumentParser


def main(config):

    init_logger(config)
    config = merge_config(config)
    # e.g. import funcs from benchmarks/resnet50/pytorch/__init__.py
    benchmark_module = importlib.import_module(
        "benchmarks." + config.case + "." + config.framework, __package__)
    # e.g. import funcs from inference_engine/nvidia/inference.py
    vendor_module = importlib.import_module("inference_engine." +
                                            config.vendor + ".inference")
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

    model_forward_result = benchmark_module.model_forward(
        model, dataloader, evaluator, config)

    logger.info(model_forward_result)

    logger.log("Model Forward End", "")
    """
    Convert model into onnx
    """
    logger.log("Export Begin",
               "Export " + config.framework + " model into .onnx")
    start = time.time()

    benchmark_module.export_model(model, config)

    duration = time.time() - start
    logger.log("Export End", str(duration) + " seconds")
    del model
    """
    Compiling backend(like tensorRT)
    """
    logger.log("Vendor Compile Begin", "Compiling With " + config.vendor)
    start = time.time()

    toolkits = vendor_module.get_inference_toolkits(config)

    duration = time.time() - start
    logger.log("Vendor Compile End", str(duration) + " seconds")
    """
    inference using engine
    """
    logger.log("Vendor Inference Begin", "")
    start = time.time()

    infer_info = benchmark_module.engine_forward(toolkits, dataloader,
                                                 evaluator, config)
    logger.info(infer_info)

    logger.log("Vendor Inference End", "")


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--perf_dir", type=str, default="", help=".")

    parser.add_argument("--data_dir", type=str, default="", help=".")

    parser.add_argument("--log_dir", type=str, default="", help=".")

    parser.add_argument("--loglevel", type=str, default='DEBUG', help=".")

    parser.add_argument("--case", type=str, default="", help=".")

    parser.add_argument("--vendor", type=str, default="", help=".")

    parser.add_argument("--framework", type=str, default="", help=".")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


if __name__ == "__main__":
    # fake, just for test
    config_from_args = parse_args()

    config_from_args.framework = config_from_args.framework.split('_')[0]
    print("config_from_args", config_from_args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    main(config_from_args)

import yaml
import os
from loguru import logger
from collections import namedtuple


def check_dup_cfg_parm(cfg, parm):

    for item in cfg.keys():
        if item in parm.keys():
            return False
    return True


def merge_vendor(cfg, vendor_cfg, vendor):
    for item in vendor_cfg.keys():
        if item not in cfg.keys():
            logger.warning("New Config Set by " + vendor + ": " + item)
            cfg[item] = vendor_cfg[item]
        else:
            if vendor_cfg[item] == cfg[item]:
                logger.error("Redundant Config Set at " + vendor + ": " + item)
                exit(1)
            logger.debug("Set " + item + " to " + str(vendor_cfg[item]))
            cfg[item] = vendor_cfg[item]


def merge_config(config):

    configuration_path = config.perf_dir + "/configs/" + config.case + "/configurations.yaml"
    parameter_path = config.perf_dir + "/configs/" + config.case + "/parameters.yaml"
    vendor_cfg_path = config.perf_dir + "/configs/" + config.case + "/vendor_config/" + config.vendor + "_configurations.yaml"

    configuration = yaml.safe_load(open(configuration_path))
    if configuration is None:
        configuration = {}
    parameter = yaml.safe_load(open(parameter_path))
    if parameter is None:
        parameter = {}
    vendor_cfg = {}
    if os.path.exists(vendor_cfg_path):
        vendor_cfg = yaml.safe_load(open(vendor_cfg_path))

    merged_data = {}
    merged_data["perf_dir"] = config.perf_dir
    merged_data["data_dir"] = config.data_dir
    merged_data["log_dir"] = config.log_dir
    merged_data["vendor"] = config.vendor
    merged_data["case"] = config.case
    merged_data["framework"] = config.framework

    filename = config.case + "_bs" + str(configuration["batch_size"])
    filename = filename + "_" + str(config.framework)
    filename = filename + "_fp16" + str(configuration["fp16"])
    filename = "onnxs/" + filename + ".onnx"
    merged_data["onnx_path"] = filename

    if not check_dup_cfg_parm(configuration, parameter):
        logger.error(
            "Duplicated terms in configurations.yaml and parameters.yaml")
        exit(1)
    merge_vendor(configuration, vendor_cfg, config.vendor)

    for item in configuration.keys():
        if item in merged_data.keys():
            logger.error(
                "Duplicated terms in configurations.yaml and host.yaml")
            exit(1)
        merged_data[item] = configuration[item]

    for item in parameter.keys():
        if item in merged_data.keys():
            logger.error("Duplicated terms in parameters.yaml and host.yaml")
            exit(1)
        merged_data[item] = parameter[item]

    Config = namedtuple("Config", merged_data.keys())
    unmutable_config = Config(**merged_data)
    return unmutable_config

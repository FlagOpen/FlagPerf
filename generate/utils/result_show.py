import os
from .config_analysize import read_last_n_lines, read_yaml_file
from loguru import logger
import re
import gc


def resultshow():

    config_path = "host.yaml"
    config_dict = read_yaml_file(config_path)

    currentdir = os.path.dirname(os.path.abspath(__file__))
    grandparent_path = os.path.dirname(currentdir)

    path = grandparent_path + "/tasks/" + config_dict[
        "vendor"] + "/" + config_dict["engine"] + "/" + config_dict[
            "chip"] + ".yaml"
    gpu_config = read_yaml_file(path)

    infer_api = config_dict["engine"]

    log_path = config_dict["log_path"] + infer_api
    filenames = os.listdir(log_path)

    token_per_s = 0
    newly_tokens = 0
    duration = 0
    ttft = 0
    total_tokens = 0
    rougeone = 0
    rougeL = 0
    rougetwo = 0
    throughput = 0

    for filename in filenames:
        if "ttft" in filename:
            data = read_last_n_lines(log_path + "/" + filename, 10)
            for i in data:
                text = re.findall(r'\b[a-zA-Z]+\b', i)
                numbers = re.findall(r'\d+', i)
                if "TTFT" in text:
                    ttft = float(numbers[-2] + "." + numbers[-1])
                elif "parameter" in text:
                    parameters = float(numbers[-2] + "." + numbers[-1])
        elif "throughput" in filename:
            data = read_last_n_lines(log_path + "/" + filename, 10)
            for i in data:
                text = re.findall(r'\b[a-zA-Z]+\b', i)
                numbers = re.findall(r'\d+', i)
                if "newly" in text:
                    newly_tokens = float(numbers[-2] + "." + numbers[-1])
                elif "total" in text:
                    total_tokens = float(numbers[-2] + "." + numbers[-1])
                elif "duration" in text:
                    duration = float(numbers[-2] + "." + numbers[-1])
                elif "token" in text and "s" in text:
                    token_per_s = float(numbers[-2] + "." + numbers[-1])
                elif "rougeone" in text:
                    rougeone = float(numbers[-2] + "." + numbers[-1])
                elif "rougetwo" in text:
                    rougetwo = float(numbers[-2] + "." + numbers[-1])
                elif "rougeL" in text:
                    rougeL = float(numbers[-2] + "." + numbers[-1])
    tflops = gpu_config["TFLOPS_FP16"]
    MFU = (token_per_s * 2 * parameters) / (tflops * 1e12 *
                                            config_dict["nproc_per_node"])
    if duration > ttft:
        throughput = newly_tokens / (duration - ttft)
    else:
        throughput = newly_tokens / (duration)
    return ttft, newly_tokens, total_tokens, duration, token_per_s, rougeone, rougetwo, rougeL, MFU, throughput

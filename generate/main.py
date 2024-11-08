import multiprocessing.process
from utils.config_analysize import read_last_n_lines, read_yaml_file
import subprocess
from utils.result_show import resultshow
from loguru import logger
import os, sys
import time
import multiprocessing

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../")))

config_dict = read_yaml_file("host.yaml")
infer_api = config_dict["engine"]
config_path = config_dict["config_path"]
gpuname = config_dict["vendor"]
with open('output.txt', 'w') as f:
    run_path = "result/" + "" + infer_api + "_" + gpuname + "_" + time.strftime(
        "%Y%m%d%H%M%S")
    utils_path = "../inference/docker_images/" + gpuname + "/" + gpuname + "_monitor.py"
    monitor_process = subprocess.Popen(
        ["python", utils_path, "-o", "restart", "-l", run_path],
        stdout=f,
        stderr=f)
    logger.info("Starting...")
    time.sleep(4)
    logger.info("Starting to run inference...")
    logger.info("waiting for monitor....")
    time.sleep(4)
    logger.info("Start measuring TTFT")
    subprocess.Popen([
        "python", "TTFT/" + gpuname + "/" + infer_api + "/" + "ttft.py",
        config_path
    ],
                     stdout=f,
                     stderr=f).wait()
    time.sleep(4)
    logger.info("Start measuring Throughput")
    subprocess.Popen([
        "python", "Throughput/" + gpuname + "/" + infer_api + "/" +
        "throughput.py", config_path
    ],
                     stdout=f,
                     stderr=f).wait()
    logger.info("Finished")
    time.sleep(4)
    logger.info("Starting to show results")
    ttft, newly_tokens, total_tokens, duration, tps, rougeone, rougetwo, rougeL, MFU, throughput = resultshow(
    )

    logger.info(f"TTFT:{ttft}")
    logger.info(f"Throughput:{throughput}")
    logger.info(f"Tps:{tps}")
    logger.info(f"MFU:{MFU}")
    logger.info(f"ROUGE1:{rougeone}")
    logger.info(f"ROUGE2:{rougetwo}")
    logger.info(f"ROUGEL:{rougeL}")

    time.sleep(5)
    monitor_process = subprocess.Popen(
        ["python", utils_path, "-o", "stop", "-l", run_path],
        stdout=f,
        stderr=f)

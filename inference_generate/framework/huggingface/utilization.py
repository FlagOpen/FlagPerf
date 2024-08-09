import subprocess
import time
from loguru import logger
from config_analysis import read_last_n_lines,read_yaml_file
def get_gpu_utilization(logfile,stopevent):
    model_config=read_yaml_file("./modelconfig.yaml")
    logger.remove()
    logger.add(logfile,format="{time} {level} {message}",level="INFO",mode='w')
    while not stopevent.is_set():
        result=subprocess.run(model_config["UTIL_ANALYSIS"],capture_output=True,text=True)
        utilization= result.stdout.split('\n')[:-1]
        logger.info(f"GPU utilization: {utilization}")
        logger.info(f"--------------------------------------------------------------------------")
        time.sleep(3) # wait for 1 second before checking again
    logger.info("utilization monitoring stopped")
    
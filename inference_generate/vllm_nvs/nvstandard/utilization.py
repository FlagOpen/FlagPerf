import subprocess
import time
from loguru import logger
def get_gpu_utilization(logfile,stopevent):
    logger.remove()
    logger.add(logfile,format="{time} {level} {message}",level="INFO",mode='w')
    while not stopevent.is_set():
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        utilization= result.stdout.split('\n')[:-1]
        logger.info(f"GPU utilization: {utilization}")
        logger.info(f"--------------------------------------------------------------------------")
        time.sleep(3) # wait for 1 second before checking again
    logger.info("utilization monitoring stopped")
    
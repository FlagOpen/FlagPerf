import time
from loguru import logger 
import subprocess
from config_analysis import read_last_n_lines,read_yaml_file
def get_gpu_power():
    model_config=read_yaml_file("./modelconfig.yaml")
    result=subprocess.run(model_config["POWER_ANALYSIS"],capture_output=True,text=True)
    power_draws=result.stdout.split('\n')
    return [float(p) for p in power_draws if p]
def monitor_gpu_power(logfile,stop_event):
    powers=[]
    logger.remove()
    logger.add(logfile,format="{time} {level} {message}",level='INFO',rotation='1 day',mode='w')
    while not stop_event.is_set():
        power=get_gpu_power()
        powers.append(power)
        logger.info(f"GPU power usage: {power} W")
        logger.info(f"--------------------------------------------------------------------------")
        time.sleep(3) # wait for 1 second before checking again
    logger.info("Power monitoring stopped")
        
    # pynvml.nvmlInit()# initialize the NVML library
    # logger.remove()
    # logger.add(logfile,format="{time} {level} {message}",level='INFO',rotation='1 day')
    # deviceCount = pynvml.nvmlDeviceGetCount() # get the number of GPUs
    # while not stop_event.is_set():
    #     for i in range(deviceCount):
    #         handle = pynvml.nvmlDeviceGetHandleByIndex(i) # get the handle for the GPU
    #         power = pynvml.nvmlDeviceGetPowerUsage(handle)/1000 # get the power usage in milliwatts
    #         logger.info(f"GPU {i} power usage: {power} W")
    #     logger.info(f"--------------------------------------------------------------------------")
    #     time.sleep(3) # wait for 1 second before checking again
    # logger.info("Power monitoring stopped")

    
        
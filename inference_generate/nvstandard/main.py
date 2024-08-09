import subprocess
from loguru import logger
import pandas as pd
import numpy as np
from config_analysize import read_yaml_file,read_last_n_lines
import re
with open('output.txt', 'w') as f:
    logger.info("Starting the program")
    subprocess.Popen(["nsys","profile","--stat=true","--force-overwrite=true","--output","nsys_profile","python","timemonitor.py"],stdout=f,stderr=f).wait()
    subprocess.run(["nsys","stats","-f","csv","--force-overwrite","-o","./result/file.csv","nsys_profile.nsys-rep"],stdout=f,stderr=f)
    logger.info("Count END")
    logger.info("Compute begin")
    df=pd.read_csv("/home/daliwang/FlagPerf/newinference/inference_generate/nvstandard/result/file.csv_cuda_gpu_kern_sum.csv")
    total_time=df["Total Time (ns)"].sum()*1e-9
    logger.info(f"kernel time:{total_time}s")
    yaml_data = read_yaml_file('/home/daliwang/FlagPerf/newinference/inference_generate/config/GPU_config.yaml')
    actual_flop=yaml_data['GPU_FP16TC']*float(total_time*1000*1e9)
    logger.info(f"actual flops:{actual_flop}")
    data=read_last_n_lines('/home/daliwang/FlagPerf/newinference/inference_generate/nvstandard/task_time.log',8)
    MFU=0.0
    for i in data:
        if "All Flops" in i:
            numbers=re.findall(r'\d+', i)
            MFU+=float(round(float(str(numbers[-2])+"."+str(numbers[-1]))/actual_flop,4))*100
            continue
        text=re.findall(r'\b[a-zA-Z]+\b', i)
        numbers=re.findall(r'\d+', i)
        if "END" in text:
            break
        elif len(text)==1:
            continue
        else:
            metrics=""
            for i in text:
                if i=="INFO":
                    continue
                metrics+=i+" "
            metrics+=":"
            ans=round(float(str(numbers[-2])+"."+str(numbers[-1])),4)
            if "time" in metrics:
                logger.info(f"{metrics}{ans}s")
                continue
            logger.info(f"{metrics}{ans}")
    MFU=str(MFU)+"%"
    logger.info(f"MFU: {MFU}")
    
    
    
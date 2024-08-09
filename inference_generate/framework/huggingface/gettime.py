import pandas as pd
from tqdm import tqdm
import time
from loguru import logger
from thop import profile
import subprocess
import multiprocessing
import config_analysis
import importlib
def time_monitor(logfile,stop_event):
    model_config="./modelconfig.yaml"
    df2=pd.read_csv(model_config["DATAPATH"])
    local_model_path=model_config["MODELPATH"]
    tasks=df2['dialogue'].tolist()
    frame=importlib.import_module(model_config["FRAMEWORK"])
    device=frame.device(model_config["DEVICE"] if frame.cuda.is_available() else "cpu")
    infer_api=importlib.import_module(model_config["INFER_API"])
    tokenizer = infer_api.AutoTokenizer.from_pretrained(local_model_path)
    model=infer_api.AutoModelForCausalLM.from_pretrained(local_model_path,device_map="auto").eval()
    #model=Accelerator().prepare_model(model)
    # if frame.cuda.device_count() >1:
    #     n_gpu=frame.cuda.device_count()
    #     model = frame.nn.DataParallel(model,device_ids=list(range(n_gpu)))
    time_all=0
    token_sum=0
    warmup_before=0
    flops_sum=0
    logger.remove()
    logger.add(logfile,format="{time} {level} {message}",level='INFO',rotation='1 day',mode='w')
    for i in tqdm(range(10),desc="inference"):
        with frame.no_grad():
            timest=time.time()
            input_text=tasks[i]
            logger.info(f"--------------------------------------------------------------------")
            logger.info(f"input lenth is: {len(input_text)}")
            #print("input lenth is: ", len(input_text))
            inputs=tokenizer([input_text+"Please generate a summary about it:"],return_tensors="pt")
            inputs=inputs.to(device)
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1,  # 控制生成的额外令牌数
                num_return_sequences=1,  # 返回序列数量
                no_repeat_ngram_size=2,  # 避免重复的n-gram
                early_stopping=True , # 如果达到良好的结束点则停止生成
                attention_mask=inputs.attention_mask,
                num_beams=5,
                top_k=50
            )
            warmup_time=time.time()-timest
            logger.info(f"warmup time is: {warmup_time}")#测量首字热身时间
            warmup_before+=warmup_time
            timest=time.time()
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=4096,  # 控制生成的额外令牌数
                num_return_sequences=1,  # 返回序列数量
                no_repeat_ngram_size=2,  # 避免重复的n-gram
                early_stopping=True , # 如果达到良好的结束点则停止生成
                attention_mask=inputs.attention_mask,
                num_beams=5,
                top_k=50
            )
            duration=time.time()-timest
            logger.info(f"duration is: {duration}")#测量wall clock time
            flops,params=profile(model,(inputs.input_ids,))
            flops_sum+=flops
            logger.info(f"model flops is: {flops}")#测量flops
            logger.info(f"model params is: {params}")#测量params
            # ncu_command=["ncu","--metrics","smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",f"python -c 'import frame; from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained(\"{local_model_path}\"); model.eval();model(\"{inputs}\");'"]
            # result=subprocess.run(ncu_command, capture_output=True, text=True)
            # ncu_output=result.stdout
            # logger.info(f"actual flops is: {ncu_output}")#测量实际flops
            time_all+=duration
            token_sum+=len(outputs[0])
            generated=tokenizer.decode(outputs[0], skip_special_tokens=True)
            #print("generated lenth is: ", len(generated))
            logger.info(f"generated lenth is: {len(generated)}")
            logger.info(f"--------------------------------------------------------------------")
    #print(time_all)
    logger.info(f"total time is: {time_all}")
    #print(time_all/60)
    logger.info(f"Average warm up time per task is: {warmup_before/60}")
    logger.info(f"Task/s is: {float(60/time_all)}")
    logger.info(f"Average time per task is: {time_all/60}")
    #print(float(token_sum/time_all))
    logger.info(f"Token/s is: {float(token_sum/time_all)}")
    logger.info(f"All Flops is {float(flops_sum)}")
    logger.info(f"END")
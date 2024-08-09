from power_monitor import monitor_gpu_power
from loguru import logger
from utilization import get_gpu_utilization
import subprocess
import multiprocessing
from gettime import time_monitor
if True:
    stop_event=multiprocessing.Event()
    logfile_power='gpu_power_log.log'
    logfile_time="task_time.log"
    logfile_utilization="utilization.log"
    process_C=multiprocessing.Process(target=get_gpu_utilization, args=(logfile_utilization,stop_event))
    process_C.start()
    process_B=multiprocessing.Process(target=monitor_gpu_power, args=(logfile_power,stop_event))
    process_B.start()
    # with open("kernel_time_monitor.txt", "w") as f:
    #     process=subprocess.Popen(["nsys","profile","--stat=true","python","main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     stdout,stderr=process.communicate()
    #     f.write(stdout.decode("utf-8"))
    #process_A=multiprocessing.Process(target=time_monitor, args=(logfile_time,stop_event))
    time_monitor(logfile_time,stop_event)
    stop_event.set()
    process_B.join()
    process_C.join()
    
    
    
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
# input_text="buddy,can you speak English?"
# inputs=tokenizer([input_text],return_tensors="pt")
# outputs = model.generate(
#     inputs.input_ids,
#     max_new_tokens=10,  # 控制生成的额外令牌数
#     num_return_sequences=1,  # 返回序列数量
#     no_repeat_ngram_size=2,  # 避免重复的n-gram
#     early_stopping=True , # 如果达到良好的结束点则停止生成
#     attention_mask=inputs.attention_mask,
#     num_beams=5,
#     top_k=50
# )
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("----------------------------------------------------------------")
# print(input_text)
# print("Generated Text: ", generated_text)
# print("----------------------------------------------------------------")

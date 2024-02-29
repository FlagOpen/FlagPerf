import subprocess
import threading
import time
from argparse import ArgumentParser
import os


def parse_args():
    parser = ArgumentParser(description="aquila_monitor")
    parser.add_argument("--ip", type=str, required=True)
    args=parser.parse_args()
    return args


def run_cmd(cmd, interval, outputstream):
    while True:
        subprocess.Popen(cmd, shell=True, stdout=outputstream, stderr=subprocess.STDOUT)
        time.sleep(interval)


def main():
    args = parse_args()
    ip = args.ip.replace('.','_')
    log_dir = "./" + "monitor34" + "/" +ip + "/"
    os.mkdir(log_dir)

    cmd = r"echo NODE " + args.ip + ";"
    cmd = cmd + r"echo ;"

    cmd = cmd + r"echo OS version:;"
    cmd = cmd + r"cat /etc/issue | head -n1 | awk '{print $1, $2, $3}';"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo OS Kernel version:;"
    cmd = cmd + r"uname -r;"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo Accelerator Model:;"
    cmd = cmd + r"rocm-smi --showhw;"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo Accelerator Driver version:;"
    cmd = cmd + r"rocm-smi --showdriverversion | grep 'Driver version' | awk '{print $3}';"
    cmd = cmd + r"echo ;"
    
    
    sys_fn = log_dir + "sys_info.log.txt"
    with open(sys_fn, "w") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
        
    threads = []
    
    mem_cmd = "date;free -g|grep -i mem|awk '{print $3/$2}';echo \"\""
    mem_file = open(log_dir + "mem.log.txt", "w")
    mem_thread = threading.Thread(target=run_cmd, args=(mem_cmd, 5, mem_file))
    threads.append(mem_thread)
    
    cpu_cmd = "date;mpstat -P ALL 1 1|grep -v Average|grep all|awk '{print (100-$NF)/100}';echo \"\""
    cpu_file = open(log_dir + "cpu.log.txt", "w")
    cpu_thread = threading.Thread(target=run_cmd, args=(cpu_cmd, 5, cpu_file))
    threads.append(cpu_thread)
    
    pwr_cmd = "date;ipmitool sdr list|grep -i Watts|grep -E 'Node_Pwr_N1'|grep -oP '\d+ Watts';echo \"\""
    pwr_file = open(log_dir + "pwr.log.txt", "w")
    pwr_thread = threading.Thread(target=run_cmd, args=(pwr_cmd, 120, pwr_file))
    threads.append(pwr_thread)
    
    gpu_cmd = "date;rocm-smi |grep 'manual' |awk '{print $2,$3,$7*0.16'GB','16GB',$8}';echo \"\""
    gpu_file = open(log_dir + "gpu.log.txt", "w")
    gpu_thread = threading.Thread(target=run_cmd, args=(gpu_cmd, 5, gpu_file))
    threads.append(gpu_thread)

    for thread in threads:
        thread.start()

    while True:
        time.sleep(0.1)

    print('exit')


if __name__ == '__main__':
    main()



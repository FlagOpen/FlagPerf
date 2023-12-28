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
    log_dir = "./" + ip + "/"
    os.mkdir(log_dir)

    cmd = r"echo NODE " + args.ip + ";"
    cmd = cmd + r"echo ;"

    cmd = cmd + r"echo OS version:;"
    cmd = cmd + r"cat /etc/issue | head -n1 | awk '{print $1, $2, $3}';"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo OS Kernel version:;"
    cmd = cmd + r"uname -r;"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo Hardware Model:;"
    cmd = cmd + r"sudo dmidecode | grep -A9 'System Information' | tail -n +2 | sed 's/^[ \t]*//';"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo Accelerator Model:;"
    cmd = cmd + r"mthreads-smi -L;"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo Accelerator Driver version:;"
    cmd = cmd + r"mthreads-gmi | grep 'Driver Version' | awk -F ':' '{print $3}';"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo Docker version:;"
    cmd = cmd + r"docker -v"
    
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
    
    pwr_cmd = "date;ipmitool sdr list|grep -i Watts|awk 'BEGIN{FS = \"|\"}{for (f=1; f <= NF; f+=1) {if ($f ~ /Watts/) {print $f}}}'|awk '{print $1}'|sort -n -r|head -n1;echo \"\""
    pwr_file = open(log_dir + "pwr.log.txt", "w")
    pwr_thread = threading.Thread(target=run_cmd, args=(pwr_cmd, 120, pwr_file))
    threads.append(pwr_thread)
    
    gpu_cmd = "date;mthreads-gmi -q | grep -E 'GPU Current Temp|Power Draw|Used|Total|Gpu' | awk -F ': *' '/GPU Current Temp|Power Draw|Used|Total|Gpu/ { values[(NR-1)%5+1] = $2; } NR % 5 == 0 { print values[4], values[5], values[2], values[1], values[3]; }'"
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



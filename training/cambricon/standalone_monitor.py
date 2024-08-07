import subprocess
import threading
import time
from argparse import ArgumentParser
import os

def parse_args():
    parser = ArgumentParser(description="aquila_monitor")
    parser.add_argument("--ip", type=str, help="use hostname -i to find ip address", required=True)
    args=parser.parse_args()
    return args

def run_cmd(cmd, interval, outputstream):
    while True:
        subprocess.Popen(cmd, shell=True,executable="/bin/bash", stdout=outputstream, stderr=subprocess.STDOUT)
        time.sleep(interval)

def run_pwr_cmd(cmd, interval, file_name):
    with open(file_name,"w") as f:
        while True:
            process = subprocess.Popen(cmd, shell=True,executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _out, error = process.communicate()
            _out = _out.decode()
            out = _out.strip().split("\n")
            _hex = out[1].split(" ")
            hex_string = _hex[2] + _hex[1]
            int_string = str(int(hex_string,16))
            ret = out[0] + "\n" + int_string
            f.write(ret + "\n" + " " + "\n")
            f.flush()
            time.sleep(interval)

# get full machine power using bmp ip address
# but not valid for every machine,please pay attention it !
# you should calculate your bmc ip by change this code!
def get_bmc_ip(host_ip):
    _host_ip = host_ip.split(".")
    _host_ip[1] = '4'
    return ".".join(_host_ip)

# python standalone_monitor.py --ip "`hostname -i`"
def main():
    args = parse_args()
    args.ip =  args.ip.split(" ")[0]
    print("IP:",args.ip)
    print("Running......:",args.ip)
    ip = args.ip.replace('.','_')
    log_dir = "./" + ip + "/"
    if not os.path.exists(log_dir):
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
    cmd = cmd + r"cnmon -l;"
    cmd = cmd + r"echo ;"
    
    cmd = cmd + r"echo Accelerator Driver version:;"
    cmd = cmd + r"cnmon | grep 'CNMON' | awk '{print $3}';"
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
    
    pwr_cmd = "date;ipmitool -H " + get_bmc_ip(args.ip) + " -I lanplus -U admin -P admin raw 0x3a 0x26"
    pwr_thread = threading.Thread(target=run_pwr_cmd, args=(pwr_cmd,5,log_dir + "pwr.log.txt"))
    threads.append(pwr_thread)
    
    mlu_cmd = "date; paste <(cnmon |grep 'Default') <(cnmon |grep 'MLU' | head -n -1) | awk '{print $3,$4,$5,$9,$10,$11,$25}'; echo \"\""
    mlu_file = open(log_dir + "mlu.log.txt", "w")
    mlu_thread = threading.Thread(target=run_cmd, args=(mlu_cmd, 5, mlu_file))
    threads.append(mlu_thread)

    for thread in threads:
        thread.start()

    while True:
        time.sleep(0.1)

    print('exit')

if __name__ == '__main__':
    main()

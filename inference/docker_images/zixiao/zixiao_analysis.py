def analysis_log(logpath):
    logfile = open(logpath)

    max_usage = 0.0 ## usage_mem
    max_mem = 16.0 
    for line in logfile.readlines():
        '''
        zx-smi pwr DTemp MUsed Mem
        '''
        if "zx-smi" in line:
            line = line[:-1]
            usage = line.split(" ")[3]
            usage = float(usage)*16/100
            max_usage = max(max_usage, usage)
    return round(max_usage, 2), max_mem, eval("30e12"), eval("120e12")


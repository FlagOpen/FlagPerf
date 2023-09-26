def analysis_log(logpath):
    logfile = open(logpath)

    max_usage = 0.0 ## usage_mem
    max_mem = 0.0 
    for line in logfile.readlines():
        '''
        xpu_smi temp power mem w_mem use_rate
        '''
        if "xpu_smi" in line:
            line = line[:-1]
            usage = line.split(" ")[4]
            usage = float(usage)
            max_usage = max(max_usage, usage)
            max_mem = line.split(" ")[5]
            max_mem = float(max_mem)

    return round(max_usage / 1024.0,
                 2), round(max_mem / 1024.0, 2), eval("32e12"), eval("128e12")



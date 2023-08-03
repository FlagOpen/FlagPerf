def analysis_log(logpath):
    logfile = open(logpath)

    max_usage = 0.0
    max_mem = 0.0
    for line in logfile.readlines():
        if "MiB" in line:
            usage = line.split(" ")[2]
            usage = float(usage[:-3])
            max_usage = max(max_usage, usage)
            max_mem = line.split(" ")[3]
            max_mem = float(max_mem[:-3])

    return round(max_usage / 1024.0, 2), round(max_mem / 1024.0, 2)

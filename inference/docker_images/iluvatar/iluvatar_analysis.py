def analysis_log(logpath):
    logfile = open(logpath)

    max_usage = 0.0
    max_mem = 0.0
    for line in logfile.readlines():
        if "MiB" in line:
            usage = line.split(" ")[2]
            usage = float(usage[:-3])
            max_usage = max(max_usage, usage)
            if (max_usage >= usage):
                m_mem = line.split(" ")[3]
                m_mem = float(m_mem[:-3])
                max_mem = max(max_mem, m_mem)

    return round(max_usage / 1024.0,
                 2), round(max_mem / 1024.0, 2), eval("24e12"), eval("96e12")

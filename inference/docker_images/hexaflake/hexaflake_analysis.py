import re

def analysis_log(logpath):
    logfile = open(logpath)

    max_usage = 0.0 ## usage_mem
    max_mem = 0.0
    for line in logfile.readlines():
        '''
        hxsmi pwr DTemp MUsed Mem
        '''
        if "/" in line:
            line = line[:-1]
            match = re.search(r'(\d+)MiB / (\d+)MiB', line)
            max_mem = float(match.group(2))
            usage = float(match.group(1))
            max_usage = max(max_usage, usage)
    return round(max_usage, 2), round(max_mem, 2), eval("30e12"), eval("120e12")

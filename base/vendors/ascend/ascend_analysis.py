def analysis_log(logpath, config):
    logfile = open(logpath)

    result = {"temp": {}, "power": {}, "mem":{}}
    for gpuID in range(8):
        for monitor_index in result.keys():
            result[monitor_index][gpuID] = []

    max_mem = 65536
    next_gpu_id = 0

    for line in logfile.readlines():
        if line != '\n' and ':' not in line and '-' not in line:
            result["max_mem"] = max_mem
            power = float(line.split(" ")[0])
            temp = float(line.split(" ")[1])
            mem = int(line.split(" ")[3].replace('\n', '').replace('/', ''))
            result["temp"][next_gpu_id].append(temp)
            result["power"][next_gpu_id].append(power)
            result["mem"][next_gpu_id].append(mem)
            next_gpu_id = (next_gpu_id + 1) % 8

    return result
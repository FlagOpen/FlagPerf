def analysis_log(logpath, config):
    logfile = open(logpath)

    result = {"temp": {}, "power": {}, "mem": {}}
    for gpuID in range(config.NPROC_PER_NODE):
        for monitor_index in result.keys():
            result[monitor_index][gpuID] = []

    max_mem = None
    next_gpu_id = 0

    for line in logfile.readlines():
        if "MiB" in line:
            if max_mem is None:
                max_mem = float(line.split(" ")[5])
                result["max_mem"] = max_mem
            temp = float(line.split(" ")[0][:-1])
            power = float(line.split(" ")[1])
            mem = float(line.split(" ")[3])
            result["temp"][next_gpu_id].append(temp)
            result["power"][next_gpu_id].append(power)
            result["mem"][next_gpu_id].append(mem)
            next_gpu_id = (next_gpu_id + 1) % config.NPROC_PER_NODE

    return result

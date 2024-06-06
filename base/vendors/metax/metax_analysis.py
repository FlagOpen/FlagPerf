def analysis_log(logpath, config):
    logfile = open(logpath)

    result = {"temp": {}, "power": {}, "mem": {}}
    for gpuID in range(config.NPROC_PER_NODE):
        for monitor_index in result.keys():
            result[monitor_index][gpuID] = []

    max_mem = None
    next_gpu_id = 0
    max_usage = 0.0
    for line in logfile.readlines():
        if "MiB" in line:

            if max_mem is None:
                usage_and_maxusage = line.split(" ")[2]
                result["max_mem"] = float(usage_and_maxusage.split("/")[1])
            temp_str = line.split(" ")[0]
            temp =  (float(temp_str[:-1]))
            power_str = line.split(" ")[1]
            power =  (float(power_str[:-1]))
            usage_and_maxusage = line.split(" ")[2]
            usage = float(usage_and_maxusage.split("/")[0])
            max_usage = max(max_usage, usage)
            max_mem = float(usage_and_maxusage.split("/")[1])


            print('@@@@ ')
            print(temp)
            print(power)
            print (max_mem)
            print (max_usage)
            print('end @@@@ ')
            result["temp"][next_gpu_id].append(temp)
            result["power"][next_gpu_id].append(power)
            result["mem"][next_gpu_id].append(max_usage)
            next_gpu_id = (next_gpu_id + 1) % config.NPROC_PER_NODE

    return result
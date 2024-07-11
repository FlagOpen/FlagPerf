def analysis_log(logpath, config):
    logfile = open(logpath)

    result = {"temp": {}, "power": {}, "mem": {}}
    for mluID in range(config.NPROC_PER_NODE):
        for monitor_index in result.keys():
            result[monitor_index][mluID] = []

    max_mem = None
    next_mlu_id = 0

    for line in logfile.readlines():
        if "C" in line:
            if max_mem is None:
                max_mem = float(line.split(" ")[3])
                result["max_mem"] = max_mem
            temp = float(line.split(" ")[0][:-1])
            power = float(line.split(" ")[1])
            mem = float(line.split(" ")[2])
            result["temp"][next_mlu_id].append(temp)
            result["power"][next_mlu_id].append(power)
            result["mem"][next_mlu_id].append(mem)
            next_mlu_id = (next_mlu_id + 1) % config.NPROC_PER_NODE

    return result

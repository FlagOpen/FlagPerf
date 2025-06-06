def analysis_log(logpath, config):
    logfile = open(logpath)

    result = {"temp": {}, "power": {}, "mem": {}}
    for gpuID in range(config.NPROC_PER_NODE):
        for monitor_index in result.keys():
            result[monitor_index][gpuID] = []

    max_mem = None
    next_gpu_id = 0

    for line in logfile.readlines():
        # |   0   TX8110-64GB-PCIe   11100   42C     96W / 200W  |  00000000:82:00.0  |   15146M / 65536M     0%    |  1  |
        if "TX81" in line:
            if max_mem is None:
                max_mem = float(line.split('|')[-3].strip().split('/')[-1].split('M')[0].strip())
                result["max_mem"] = max_mem
            temp  = float(line.split("C")[1][-3:].strip())
            power = float(line.split('W')[0][-4:].strip())
            mem   = float(line.split('M')[0][-6:].strip())
            result["temp"][next_gpu_id].append(temp)
            result["power"][next_gpu_id].append(power)
            result["mem"][next_gpu_id].append(mem)
            next_gpu_id = (next_gpu_id + 1) % config.NPROC_PER_NODE

    return result
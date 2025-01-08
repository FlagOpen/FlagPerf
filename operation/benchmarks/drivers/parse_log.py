# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
from loguru import logger


def parse_log_file(spectflops, mode, warmup, log_dir, save_log_path):
    with open(log_dir, 'r') as file_r, open(save_log_path, 'w') as file_w:
        lines = file_r.readlines()
    for line in lines:
        if line.startswith("[INFO]"):
            json_data = line[6:].strip()
            try:
                data = json.loads(json_data)
                op_name = data.get("op_name")
                dtype = data.get("dtype")
                results = data.get("result")
                for result in results:
                    shape_detail = result.get("shape_detail")
                    latency_base = result.get("latency_base")
                    if mode == "cpu" and warmup == 0:
                        no_warmup_latency = result.get("latency")
                    elif mode == "cpu" and warmup == 1000:
                        warmup_latency = result.get("latency")
                        raw_throughput = 1 / int(warmup_latency)
                        ctflops = result.get("tflops")
                        cfu = round(100.0 * ctflops / 1E12 / spectflops, 2)
                    elif mode == "cuda" and warmup == 1000:
                        kerneltime = result.get("latency")
                        core_throughput = 1 / int(kerneltime)
                        ktflops = result.get("tflops")
                        kfu = round(100.0 * ktflops / 1E12 / spectflops, 2)
                    else:
                        pass
                    parse_data = {
                        "op_name": op_name,
                        "dtype": dtype,
                        "mode": mode,
                        "warmup": warmup,
                        "shape_detail": shape_detail,
                        "latency_base": latency_base,
                        "no_warmup_latency": no_warmup_latency,
                        "warmup_latency": warmup_latency,
                        "raw_throughput": raw_throughput,
                        "ctflops": ctflops,
                        "cfu": cfu,
                        "kerneltime": kerneltime,
                        "core_throughput": core_throughput,
                        "kfu": kfu
                    }
                    new_line = "INFO" + str(parse_data)
                    file_w.write(new_line + '\n')
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")



# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import os
from collections import defaultdict
from loguru import logger


def parse_log_file(spectflops, mode, warmup, log_dir, result_log_path):
    log_file = os.path.join(log_dir, "result.log.txt")
    save_log_path = os.path.join(result_log_path, "result.json")
    if os.path.isfile(save_log_path):
        with open(save_log_path, 'r+', encoding='utf-8') as file_r:
            file_r_json = file_r.read()
            if file_r_json:
                res = json.loads(file_r_json)
                result_data = get_result_data(log_file, res, spectflops, mode, warmup)
                file_r.seek(0)
                file_r.write(json.dumps(result_data, ensure_ascii=False))
                file_r.truncate()
            else:
                logger.error("Contents of the file is empty！！！！")
    else:
        with open(save_log_path, 'w') as file_w:
            res = defaultdict(dict)
            result_data = get_result_data(log_file, res, spectflops, mode, warmup)
            file_w.write(json.dumps(result_data, ensure_ascii=False))


def get_result_data(log_file, res, spectflops, mode, warmup):
    # 参数说明
    # 时延：1 无预热时延 Latency-No warmup：no_warmup_latency，2 预热时延 Latency-Warmup：warmup_latency
    # 吞吐率：3 Raw-Throughput原始吞吐：raw_throughput， 4 Core-Throughput是核心吞吐：core_throughput
    # 算力：5 实际算力开销：ctflops， 6 实际算力利用率：cfu， 7 实际算力开销-内核时间：ktflops， 8 实际算力利用率-内核时间：kfu
    with open(log_file, 'r') as file_r:
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
                        if mode == "cpu" and warmup == "0":
                            no_warmup_latency = result.get("latency")
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "mode": mode,
                                "shape_detail": shape_detail,
                                "latency_base_cpu_nowarm": latency_base,
                                "no_warmup_latency": no_warmup_latency
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "cpu" and warmup == "1000":
                            warmup_latency = result.get("latency")
                            raw_throughput = 1 / float(warmup_latency)
                            ctflops = result.get("tflops")
                            if ctflops is None:
                                cfu = None
                            else:
                                cfu = round(100.0 * float(ctflops) / 1e12 / float(spectflops), 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "mode":  mode,
                                "shape_detail": shape_detail,
                                "latency_base_cpu_warm": latency_base,
                                "warmup_latency": warmup_latency,
                                "raw_throughput": raw_throughput,
                                "ctflops": ctflops,
                                "cfu": cfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "cuda" and warmup == "1000":
                            kerneltime = result.get("latency")
                            core_throughput = 1 / float(kerneltime)
                            ktflops = result.get("tflops")
                            if ktflops is None:
                                kfu = None
                            else:
                                kfu = round(100.0 * float(ktflops) / 1E12 / float(spectflops), 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "mode": mode,
                                "shape_detail": shape_detail,
                                "latency_base_cuda_warm": latency_base,
                                "kerneltime": kerneltime,
                                "core_throughput": core_throughput,
                                "ktflops": ktflops,
                                "kfu": kfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
        return res
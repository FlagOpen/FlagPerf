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
    logger.info("print into parse_log_file")
    logger.info(log_dir)
    if os.path.isfile(save_log_path):
        logger.info("print result.json is exist")
        with open(save_log_path, 'r', encoding='utf-8') as f_r:
            f_r_json = f_r.read()
            if f_r_json:
                res = json.loads(f_r_json)
                result_data = get_result_data(log_file, res, spectflops, mode, warmup)
                logger.info("print one result_data")
                logger.info(result_data)
                f_r.write(json.dumps(result_data, ensure_ascii=False))
            else:
                logger.error("Contents of the file is empty！！！！")
    else:
        logger.info("print result.json not is exist")
        with open(save_log_path, 'w') as file_w:
            res = defaultdict(dict)
            result_data = get_result_data(log_file, res, spectflops, mode, warmup)
            logger.info("print two result_data")
            logger.info(result_data)
            file_w.write(json.dumps(result_data, ensure_ascii=False))


def get_result_data(log_file, res, spectflops, mode, warmup):
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
                                "warmup": warmup,
                                "shape_detail": shape_detail,
                                "latency_base": latency_base,
                                "no_warmup_latency": no_warmup_latency
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "cpu" and warmup == "1000":
                            warmup_latency = result.get("latency")
                            raw_throughput = 1 / int(warmup_latency)
                            ctflops = result.get("tflops")
                            cfu = round(100.0 * ctflops / 1E12 / spectflops, 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "mode":  mode,
                                "warmup": warmup,
                                "shape_detail": shape_detail,
                                "latency_base": latency_base,
                                "warmup_latency": warmup_latency,
                                "raw_throughput": raw_throughput,
                                "ctflops": ctflops,
                                "cfu": cfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "cuda" and warmup == "1000":
                            kerneltime = result.get("latency")
                            core_throughput = 1 / int(kerneltime)
                            ktflops = result.get("tflops")
                            kfu = round(100.0 * ktflops / 1E12 / spectflops, 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "mode": mode,
                                "warmup": warmup,
                                "shape_detail": shape_detail,
                                "latency_base": latency_base,
                                "kerneltime": kerneltime,
                                "core_throughput": core_throughput,
                                "kfu": kfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
        return res
# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import os
import sys
import yaml
from argparse import Namespace

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../")))
OP_PATH = os.path.abspath(os.path.join(CURR_PATH, "../"))
from formatMDfile import *


def main(vendor, shm_size, chip):
    result_json_file_path = os.path.join(OP_PATH, "result/result.json")
    sava_path = os.path.join(OP_PATH, "result")
    # render_base(sava_path, vendor, shm_size, chip)
    with open(result_json_file_path, 'r') as f:
        content = json.loads(f.read())
        render(content, sava_path, vendor, shm_size, chip)


if __name__ == "__main__":
    config_path = os.path.join(OP_PATH, "configs/host.yaml")
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
        config = Namespace(**config_dict)
        cases = []
        for case in config.CASES:
            cases.append(case)
        vendor = config.VENDOR
        shm_size = config.SHM_SIZE
        for run_case in cases:
            case_name = run_case
        test_file, op, dataformat, spectflops, oplib, chip = case_name.split(":")
        main(vendor, shm_size, chip)
    print("successful !!!")
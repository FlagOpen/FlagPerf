#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''This script is called in container to prepare running environment.
'''
import os
import sys
import shutil
from argparse import ArgumentParser

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../")))
from utils import run_cmd


def parse_args():
    '''Parse args with ArgumentParser.'''
    parser = ArgumentParser("Prepare running environment in Container.")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        required=True,
                        help="Model name.")
    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="Accelerator vendor.")
    parser.add_argument("--framework",
                        type=str,
                        required=True,
                        help="AI framework.")
    parser.add_argument("--pipsource",
                        type=str,
                        default="https://pypi.tuna.tsinghua.edu.cn/simple",
                        help="pip source.")
    args = parser.parse_args()
    return args


def install_requriements(vendor, model, framework, pipsource):
    '''Install required python packages in vendor's path.'''
    # framework: DL framework, may include version info. e.g. pytorch_1.13
    root_path = os.path.abspath(os.path.join(CURR_PATH, "../"))
    benchmark_path = os.path.join(root_path, "benchmarks/")
    case_path = os.path.join(benchmark_path, model)
    framework_name = framework.split("_")[0]
    framework_path = os.path.join(case_path, framework_name)

    req_file = os.path.join(framework_path, "requirements.txt")
    
    vendor_filename = vendor + "_requirements.txt"
    vendor_req_file = os.path.join(framework_path, vendor_filename)
    if os.path.exists(vendor_req_file):
        req_file = vendor_req_file

    env_file = os.path.join(framework_path, "environment_variables.sh")  
        
    if not os.path.isfile(req_file):
        print("requirenments file ", req_file, " doesn't exist. Do nothing.")
        return 0

    pip_install_cmd = "source " + env_file + "; pip3 install -r " + req_file \
                                + " -i " + pipsource
    print(pip_install_cmd)
    ret, outs = run_cmd.run_cmd_wait(pip_install_cmd, 1200)
    print(ret, outs[0])
    return ret


def install_extensions(vendor, model, framework):
    '''Install vendor's extensions with setup.py script.'''
    root_path = os.path.abspath(os.path.join(CURR_PATH, "../"))
    vendor_path = os.path.join(root_path, "inference_engine/")
    vendor_path = os.path.join(vendor_path, vendor)
    source_path = os.path.join(vendor_path, "csrc")
    env_file = os.path.join(vendor_path, "environment_variables.sh")

    if not os.path.isdir(source_path):
        print("extensioin code ", source_path, " doesn't exist. Do nothing.")
        return 0

    sandbox_dir = os.path.join(vendor_path, 'sandbox', "extension")
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)

    cmd = "source " + env_file + "; export EXTENSION_SOURCE_DIR=" \
          + source_path + " ;" + " mkdir -p " + sandbox_dir + "; cd " \
          + sandbox_dir + "; " + sys.executable + " " + source_path \
          + "/setup.py install; " + " rm -rf " + sandbox_dir
    print(cmd)
    ret, outs = run_cmd.run_cmd_wait(cmd, 1200)
    print(ret, outs[0])
    return ret


def main():
    '''Main process of preparing environment.'''
    args = parse_args()
    ret = install_requriements(args.vendor, args.model, args.framework,
                               args.pipsource)
    if ret != 0:
        sys.exit(ret)
    ret = install_extensions(args.vendor, args.model, args.framework)
    sys.exit(ret)


if __name__ == '__main__':
    main()

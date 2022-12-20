#!/bin/bash
curr_dir=$(cd $(dirname $0) && pwd)
cp ${curr_dir}/env_args.py /opt/conda/lib/python3.8/site-packages/flagai/

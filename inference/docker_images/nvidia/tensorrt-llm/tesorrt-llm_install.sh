#!/bin/bash

pip3 install --extra-index-url https://pypi.nvidia.com tensorrt-llm==0.10.0.dev2024042300
if [ $? -ne 0 ]; then
    echo "download tensorrt-llm failed..."
    exit 1
fi
echo "download tensorrt-llm succeeded..."
#!/bin/bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License")

# NVIDIA A100 环境变量配置

# 禁用 TF32 以确保 FP32 精度
export NVIDIA_TF32_OVERRIDE=0

# 优化 CUDA 设置
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置显存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Environment variables set for NVIDIA A100 Vector computation test"

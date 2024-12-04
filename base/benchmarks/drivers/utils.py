# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch

# mthreads torch_musa import
try:
    import torch_musa
except ImportError:
    pass

def set_ieee_float32(vendor):
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = False
    elif "cambricon" in vendor:
        torch.backends.mlu.matmul.allow_tf32 = False
        torch.backends.cnnl.allow_tf32 = False
    elif "mthreads" in vendor:
        torch.backends.mudnn.allow_tf32 = False
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def unset_ieee_float32(vendor):
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif "cambricon" in vendor:
        torch.backends.mlu.matmul.allow_tf32 = True
        torch.backends.cnnl.allow_tf32 = True
    elif "mthreads" in vendor:
        torch.backends.mudnn.allow_tf32 = True
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def host_device_sync(vendor):
    if vendor == "nvidia":
        torch.cuda.synchronize()
    elif "mthreads" in vendor:
        torch.musa.synchronize()
    else:
        print("unspecified vendor {}, using default pytorch \"torch.cuda.synchronize\"".format(vendor))
        torch.cuda.synchronize()


def multi_device_sync(vendor):
    if vendor == "nvidia":
        torch.distributed.barrier()
    elif "mthreads" in vendor:
        torch.distributed.barrier()
    else:
        print("unspecified vendor {}, using default pytorch \"torch.distributed.barrier\"".format(vendor))
        torch.distributed.barrier()
        

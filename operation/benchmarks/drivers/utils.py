# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch


def adapt_torch(vendor):
    if vendor == "nvidia":
        print("nvidia does nothing")
    elif vendor == "cambricon":
        from torch_mlu.utils.model_transfer import transfer
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def set_ieee_float32(vendor):
    adapt_torch(vendor)
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def unset_ieee_float32(vendor):
    adapt_torch(vendor)
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def host_device_sync(vendor):
    adapt_torch(vendor)
    if vendor == "nvidia":
        torch.cuda.synchronize()
    else:
        print(
            "unspecified vendor {}, using default pytorch \"torch.cuda.synchronize\""
            .format(vendor))
        torch.cuda.synchronize()


def multi_device_sync(vendor):
    adapt_torch(vendor)
    if vendor == "nvidia":
        torch.distributed.barrier()
    else:
        print(
            "unspecified vendor {}, using default pytorch \"torch.distributed.barrier\""
            .format(vendor))
        torch.distributed.barrier()

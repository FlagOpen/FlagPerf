import torch


def set_ieee_float32(vendor):
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def unset_ieee_float32(vendor):
    if vendor == "nvidia":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("unspecified vendor {}, do nothing".format(vendor))


def host_device_sync(vendor):
    if vendor == "nvidia":
        torch.cuda.synchronize()
    else:
        print("unspecified vendor {}, using default pytorch \"torch.cuda.synchronize\"".format(vendor))
        torch.cuda.synchronize()


def multi_device_sync(vendor):
    if vendor == "nvidia":
        torch.distributed.barrier()
    else:
        print("unspecified vendor {}, using default pytorch \"torch.distributed.barrier\"".format(vendor))
        torch.distributed.barrier()
        

def get_memory_capacity(vendor, rank):
    if vendor == "nvidia":
        return torch.cuda.get_device_properties(rank).total_memory
    else:
        print("unspecified vendor {}, return -1.0".format(vendor))
        return -1.0
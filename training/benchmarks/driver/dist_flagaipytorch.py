# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch


#def get_local_rank():
#    return int(os.environ["LOCAL_RANK"])

def get_device():
#    """
#    Return device string, e.g. cuda:0.
#    TODO: Support other accelerators.
#    """
#    local_rank = int(os.environ["LOCAL_RANK"])
#    return "cuda:"+str(local_rank)
    return "cuda"

def get_ndevice():
    return int(os.environ["N_DEVICE"])

def get_wrold_size():
    return int(os.environ["WORLD_SIZE"])

def init_dist_training_env(config):
    ''' TODO: Support other accelarators.  '''
    if config.local_rank == -1:
        config.device = torch.device("cuda")
        config.n_device = torch.cuda.device_count()
    else:
        torch.cuda.set_device(config.local_rank)
        config.device = torch.device("cuda", config.local_rank)
        host_addr_full = 'tcp://' + os.environ[
            "MASTER_ADDR"] + ':' + os.environ["MASTER_PORT"]
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend=config.dist_backend,
                                             init_method=host_addr_full,
                                             rank=rank,
                                             world_size=world_size)
        config.n_device = torch.distributed.get_world_size()
    return

#def is_main_process():
#    return int(os.environ["LOCAL_RANK"]) == 0

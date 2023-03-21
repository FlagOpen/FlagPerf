CPM_TORCH_DEMO_A100_1X1 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x1",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 1,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}

CPM_TORCH_DEMO_A100_1X2 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x2",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 2,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}

CPM_TORCH_DEMO_A100_1X4 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x4",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 4,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}

CPM_TORCH_DEMO_A100_1X8 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x1x8",
    "repeat": 1,
    "nnodes": 1,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}

CPM_TORCH_DEMO_A100_2X8 = {
    "model": "cpm",
    "framework": "pytorch",
    "config": "config_A100x2x8",
    "repeat": 1,
    "nnodes": 2,
    "nproc": 8,
    "data_dir_host": "/home/datasets_ckpt/cpm/train/",
    "data_dir_container": "/mnt/data/cpm/train/",
}
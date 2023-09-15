import subprocess
import os
from datetime import datetime

# 设置OMP_NUM_THREADS环境变量为1
os.environ["OMP_NUM_THREADS"] = "1"

# 获取当前时间的字符串表示（格式化）
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 构建新的文件名
filename = f"log_{current_time}.txt"

# 构建完整的命令
command = [
    "python", "-m", "torch.distributed.launch",
    "--nproc_per_node", "8", "run_pretraining.py",
    "--data_dir=/workspace/dataset/coco/",
    "--coco_path=/workspace/dataset/coco/",
    "--extern_config_dir=/workspace/FlagPerf/training/nvidia/detr-pytorch/config",
    "--extern_module_dir=/workspace/FlagPerf/training/nvidia/detr-pytorch/extern",
    "--extern_config_file=config_A100x1x8.py",
    "--enable_extern_config",
    "--vendor", "nvidia"
]

# 添加重定向输出到文件部分
command += ["2>&1", "|", "tee", filename]

# 执行命令
subprocess.run(" ".join(command), shell=True, check=True)

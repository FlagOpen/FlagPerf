# Common arguments

# 请在Flagperf目录下，创建MLU_FP16_FLOPS.py，
# 文件包含MLU硬件算力值，示例如下：
# MLU_FP16_FLOPS=1.0
FLOPS_DIR='../../../../'
import sys
sys.path.append(FLOPS_DIR)
from MLU_FP16_FLOPS import MLU_FP16_FLOPS
theoryflops = float(MLU_FP16_FLOPS)

# pretrain arguments
pretrain_per_device_train_batch_size = 32
pretrain_gradient_accumulation_steps = 1


# finetune arguments
finetune_per_device_train_batch_size = 16
finetune_gradient_accumulation_steps = 1
output_dir_finetune = "Output/checkpoints_finetune/llava-v1.5-13b"

# eval arguments
mmmu_data_path = "MMMU/MMMU"
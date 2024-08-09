tokenizer_path = "llama3_70b_hf"
localbs = 1
train_steps = 300
# 请在Flagperf目录下，创建MLU_FP16_FLOPS.py，
# 文件包含MLU硬件算力值，示例如下：
# MLU_FP16_FLOPS=1.0
FLOPS_DIR='../../../../'
import sys
sys.path.append(FLOPS_DIR)
from MLU_FP16_FLOPS import MLU_FP16_FLOPS
theoryflops = float(MLU_FP16_FLOPS)
megatron_path = "/workspace/Megatron-LM" # need to be aligned with DockerFile. In NGCtorch, it's /workspace/ + Megatron-LM
tensor_parallel = 8
pipeline_parallel = 2

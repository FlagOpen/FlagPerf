seqlength = 2048
batchsize = 2
datafilename = "openwebtext_baichuan2_100M.npy"
# 请在Flagperf目录下，创建MLU_FP16_FLOPS.py，
# 文件包含MLU硬件算力值，示例如下：
# MLU_FP16_FLOPS=1.0
FLOPS_DIR='../../../../'
import sys
sys.path.append(FLOPS_DIR)
from MLU_FP16_FLOPS import MLU_FP16_FLOPS
theoryflops = float(MLU_FP16_FLOPS)
epochs = 1

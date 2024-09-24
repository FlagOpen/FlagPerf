# scale_parent must under FlagPerf/ or data_dir/, otherwise you cannot mount it into baremetal, therefore cannot use shared storage
scale_parent = "/share/project/data_dir"
scale_home = f"{scale_parent}/FlagScale/build/cambricon_MLU/FlagScale"

# this cmd should install scale at <scale_home>. <scale_home> is set by flagperf.training.benchmarks.llava1.5_7b.flagscale.run_pretraining.py
scale_download_cmd = f"cd {scale_parent}"

# NV need nothing because all requirements have been established in base docker image. vendor can do anything related here
scale_install_cmd = ""

scale_conf_dir = f"{scale_home}/examples/llama/conf"
configyaml = f"{scale_conf_dir}/config.yaml"
trainyaml = f"{scale_conf_dir}/train/train_llama3_70b_finetune.yaml"
dataset = f"{scale_parent}/SAMPLE50B/llama3/llama3_dataset"
tokenizer = f"{scale_parent}/SAMPLE50B/llama3/llama3_tokenizer"
ckpt = f"{scale_parent}/llama3_ckpt"

#cmds = {"before_start": "source /root/miniconda3/bin/activate flagscale"}
cmds = {}
# flagscale's requirements
flagscale_chip_type = "MLU"
flagscale_ssh_port = 55623

# 请在Flagperf目录下，创建MLU_FP16_FLOPS.py，
# 文件包含MLU硬件算力值，示例如下：
# MLU_FP16_FLOPS=1.0
FLOPS_DIR='../../../../'
import sys
sys.path.append(FLOPS_DIR)
from MLU_FP16_FLOPS import MLU_FP16_FLOPS
flops = float(MLU_FP16_FLOPS)

# for llama's algorithm
steps = 500

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# scale_parent must under FlagPerf/ or data_dir/, otherwise you cannot mount it into baremetal, therefore cannot use shared storage
scale_parent = "/share/project/data_dir"
scale_home = f"{scale_parent}/FlagScale/build/cambricon_MLU/FlagScale"

# this cmd should install scale at <scale_home>. <scale_home> is set by flagperf.training.benchmarks.llava1.5_7b.flagscale.run_pretraining.py
scale_download_cmd = f"cd {scale_parent}"

# NV need nothing because all requirements have been established in base docker image. vendor can do anything related here
scale_install_cmd = ""

scale_conf_dir = f"{scale_home}/examples/mixtral/conf"
configyaml = f"{scale_conf_dir}/config.yaml"
trainyaml = f"{scale_conf_dir}/train/train_mixtral_8x7b.yaml"
dataset = f"{scale_parent}/SAMPLE50B/mixtral/mixtral_dataset"
tokenizer = f"{scale_parent}/SAMPLE50B/mixtral/mixtral_tokenizer"
ckpt = f"{scale_parent}/mixtral_tp2_pp4_ep4_latest"

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

# for mixtral's algorithm
steps = 1000

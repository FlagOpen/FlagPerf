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
scale_parent = "/share/project/PUBLIC/data/FlagScale/build/kunlunxin_R300p/"
scale_home = f"{scale_parent}/FlagScale"

# this cmd should install scale at <scale_home>. <scale_home> is set by flagperf.training.benchmarks.llava1.5_7b.flagscale.run_pretraining.py
#scale_download_cmd = f"cd {scale_parent}; git clone https://github.com/FlagOpen/FlagScale.git; cd FlagScale; git checkout 085811f"
scale_download_cmd = f"cd {scale_parent}; "

# NV need nothing because all requirements have been established in base docker image. vendor can do anything related here
scale_install_cmd = ""

scale_conf_dir = f"{scale_home}/examples/llama/conf"
configyaml = f"{scale_conf_dir}/config.yaml"
trainyaml = f"{scale_conf_dir}/train/train_llama3_70b_finetune.yaml"
dataset = f"/share/project/PUBLIC/data/llama3-70b/llama3_dataset"
tokenizer = f"/share/project/PUBLIC/data/llama3-70b/llama3_tokenizer"
ckpt = f"/share/project/PUBLIC/data/llama3-70b/llama3_ckpt"


cmds = {"before_start": "source ~/.bashrc;"}


# flagscale's requirements
flagscale_chip_type = "R300p"
flagscale_ssh_port = 3702
flops = -1

# for llava's algorithm
steps = 500

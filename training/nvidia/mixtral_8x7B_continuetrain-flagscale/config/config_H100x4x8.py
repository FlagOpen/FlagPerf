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
scale_parent = "/share/project/shhh/xlcmoe"
scale_home = f"{scale_parent}/FlagScale"

# this cmd should install scale at <scale_home>. <scale_home> is set by flagperf.training.benchmarks.llava1.5_7b.flagscale.run_pretraining.py
scale_download_cmd = f"cd {scale_parent}; git clone https://github.com/FlagOpen/FlagScale.git; cd FlagScale; git checkout 237377e"

# NV need nothing because all requirements have been established in base docker image. vendor can do anything related here
scale_install_cmd = ""

scale_conf_dir = f"{scale_home}/examples/mixtral/conf"
configyaml = f"{scale_conf_dir}/config.yaml"
trainyaml = f"{scale_conf_dir}/train/train_mixtral_8x7b.yaml"
dataset = f"SAMPLE50B/mixtral/mixtral_dataset"
tokenizer = f"SAMPLE50B/mixtral/mixtral_tokenizer"
ckpt = f"mixtral_ckpt"

cmds = {"before_start": "source /root/miniconda3/bin/activate flagscale"}
# flagscale's requirements
flagscale_chip_type = "H100"
flagscale_ssh_port = 22
flops = 989E12

# for llava's algorithm
steps = 1000

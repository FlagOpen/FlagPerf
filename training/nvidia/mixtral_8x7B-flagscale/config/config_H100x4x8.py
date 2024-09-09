# scale_parent must under FlagPerf/ or data_dir/, otherwise you cannot mount it into baremetal, therefore cannot use shared storage
scale_parent = "/share/project/shhh/xlcmoe"
scale_home = f"{scale_parent}/FlagScale"

# this cmd should install scale at <scale_home>. <scale_home> is set by flagperf.training.benchmarks.llava1.5_7b.flagscale.run_pretraining.py
scale_download_cmd = f"cd {scale_parent}; git clone https://github.com/FlagOpen/FlagScale.git; cd FlagScale; git checkout a44556c"

# NV need nothing because all requirements have been established in base docker image. vendor can do anything related here
scale_install_cmd = ""

scale_conf_dir = f"{scale_home}/examples/mixtral/conf"
configyaml = f"{scale_conf_dir}/config.yaml"
trainyaml = f"{scale_conf_dir}/train/train_mixtral_8x7b.yaml"
dataset = f"SAMPLE50B/mixtral/mixtral_dataset"
tokenizer = f"SAMPLE50B/mixtral/mixtral_tokenizer"

cmds = {"before_start": "source /root/miniconda3/bin/activate flagscale"}
# flagscale's requirements
flagscale_chip_type = "H100"
flagscale_ssh_port = 22
flops = 989E12

# for llava's algorithm
steps = 30

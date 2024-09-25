# scale_parent must under FlagPerf/ or data_dir/, otherwise you cannot mount it into baremetal, therefore cannot use shared storage
scale_parent = "/metax/sfwang/FlagScale/build/metax_C500"
scale_home = f"{scale_parent}/FlagScale"

# this cmd should install scale at <scale_home>. <scale_home> is set by flagperf.training.benchmarks.llava1.5_7b.flagscale.run_pretraining.py
#scale_download_cmd = f"cd {scale_parent}; git clone https://github.com/FlagOpen/FlagScale.git; cd FlagScale; git checkout 085811f"
scale_download_cmd = f"cd {scale_parent}"

# NV need nothing because all requirements have been established in base docker image. vendor can do anything related here
scale_install_cmd = ""

# locate energon. the copy from energon_install_path to flagscale/megatron/ is done by flagperf...run_pretraining.py
energon_locate_cmd = r"pip show megatron-energon | grep Location | awk -F: '{print $2}' | xargs"

scale_conf_dir = f"{scale_home}/examples/llava/conf"
configyaml = f"{scale_conf_dir}/config.yaml"
trainyaml = f"{scale_conf_dir}/train/train_llava1.5_7b.yaml"
datasetyaml = f"{scale_home}/megatron/examples/multimodal/pretrain_dataset.yaml"
prompt = f"{scale_home}/megatron/examples/multimodal/manual_prompts.json"

cmds = {"before_start": ""}
# flagscale's requirements
flagscale_chip_type = "C500"
flagscale_ssh_port = 1234
flops = -1

# for llava's algorithm
steps = 5000
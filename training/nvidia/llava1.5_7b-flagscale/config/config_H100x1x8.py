# this cmd should install scale at <scale_home>. <scale_home> is set by flagperf.training.benchmarks.llava1.5_7b.flagscale.run_pretraining.py
scale_download_cmd = "cd <scale_home>; git clone https://github.com/FlagOpen/FlagScale.git; cd FlagScale; git checkout 604f79b"

# NV need nothing because all requirements have been established in base docker image. vendor can do anything related here
scale_install_cmd = ""

# locate energon. the copy from energon_install_path to flagscale/megatron/ is done by flagperf...run_pretraining.py
energon_locate_cmd = r"pip show megatron-energon | grep Location | awk -F: '{print $2}' | xargs"

scale_conf_dir = "examples/llava/conf"
configyaml = "config.yaml"
trainyaml = "train/train_llava1.5_7b.yaml"
datasetyaml = "megatron/examples/multimodal/pretrain_dataset.yaml"
flagscale_chip_type = "H100"


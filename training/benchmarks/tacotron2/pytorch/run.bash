export XPU_VISIBLE_DEVICES=2 
python run_pretraining.py \
    --vendor kunlunxin \
    --extern_config_dir=/fp/toremove/FlagPerf/training/kunlunxin/tacotron2-pytorch/config \
    --extern_config_file=config_common.py \
    --enable_extern_config \
    --extern_module_dir=/fp/toremove/FlagPerf/training/kunlunxin/tacotron2-pytorch/extern \
    
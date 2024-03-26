# Common arguments
theoryflops = 312000000000000.0
model_name_or_path = "/root/llava1.5_pretrain/data/LLaVA-Pretrain/checkpoints/vicuna-13b-v1.5"
vision_tower = "/root/llava1.5_pretrain/data/LLaVA-Pretrain/openai/clip-vit-large-patch14-336"

# pretrain arguments
pretrain_data_path = "/root/llava1.5_pretrain/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
pretrain_image_folder = "/root/llava1.5_pretrain/data/LLaVA-Pretrain/images"
pretrain_mm_mlp_adapter = "/home/LLaVA/LLaVA-main/checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin"

# finetune arguments
finetune_data_path = "/root/llava1.5_finetune/llava_v1_5_mix665k.json"
finetune_image_folder = "/root/llava1.5_finetune/data"
output_dir_finetune = "./checkpoints_finetune/llava-v1.5-13b"

# eval arguments
mmmu_data_path = "/raid/dataset/MMMU/MMMU"
mmmu_config_path = "/home/FlagPerf/training/benchmarks/llava1.5_13b/deepspeed/config/llava1.5.yaml"
mmmu_output_path = "/home/FlagPerf/training/benchmarks/llava1.5_13b/deepspeed/output"
mmmu_answer_path = "/home/FlagPerf/training/benchmarks/llava1.5_13b/deepspeed/config/answer_dict_val.json"

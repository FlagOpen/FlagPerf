# Common arguments
theoryflops = 312000000000000.0
model_name_or_path = "LLaVA-Pretrain/checkpoints/vicuna-13b-v1.5"
vision_tower = "LLaVA-Pretrain/checkpoints/openai/clip-vit-large-patch14-336"

# pretrain arguments
pretrain_data_path = "LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
pretrain_image_folder = "LLaVA-Pretrain/images"
pretrain_mm_mlp_adapter = "Output/checkpoints_pretrain/llava-v1.5-13b/mm_projector.bin"
output_dir_pretrain = "Output/checkpoints_pretrain/llava-v1.5-13b"

# finetune arguments
finetune_data_path = "LLaVA-Finetune/llava_v1_5_mix665k.json"
finetune_image_folder = "LLaVA-Finetune/data"
output_dir_finetune = "Output/checkpoints_finetune/llava-v1.5-13b"

# eval arguments
mmmu_data_path = "MMMU/MMMU"

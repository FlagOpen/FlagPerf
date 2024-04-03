# Common arguments
theoryflops = 312000000000000.0
model_name_or_path = "LLaVA-Pretrain/checkpoints/vicuna-7b-v1.5"
vision_tower = "LLaVA-Pretrain/checkpoints/openai/clip-vit-large-patch14-336"
model_max_length = 2048

# pretrain arguments
pretrain_data_path = "LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
pretrain_image_folder = "LLaVA-Pretrain/images"
output_dir_pretrain = "Output/checkpoints_pretrain/llava-v1.5-7b"
pretrain_per_device_train_batch_size = 32
pretrain_gradient_accumulation_steps = 1

# finetune arguments
pretrain_mm_mlp_adapter = "Output/checkpoints_pretrain/llava-v1.5-7b//mm_projector.bin"
finetune_data_path = "LLaVA-Finetune/llava_v1_5_mix665k.json"
finetune_image_folder = "LLaVA-Finetune/data"
output_dir_finetune = "Output/checkpoints_finetune/llava-v1.5-7b"
finetune_per_device_train_batch_size = 16
finetune_gradient_accumulation_steps = 1

# eval arguments
mmmu_data_path = "MMMU/MMMU"

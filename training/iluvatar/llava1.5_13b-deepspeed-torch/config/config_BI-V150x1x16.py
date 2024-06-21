# Common arguments
theoryflops = 192000000000000.0

# pretrain arguments
pretrain_per_device_train_batch_size = 32
pretrain_gradient_accumulation_steps = 1


# finetune arguments
finetune_per_device_train_batch_size = 16
finetune_gradient_accumulation_steps = 1
output_dir_finetune = "Output/checkpoints_finetune/llava-v1.5-13b"

# eval arguments
mmmu_data_path = "MMMU/MMMU"

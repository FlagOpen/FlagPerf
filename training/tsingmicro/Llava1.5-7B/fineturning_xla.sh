# !/bin/sh
   clear
   export TF_CPP_MIN_LOG_LEVEL=0
   export TF_CPP_LOG_THREAD_ID=1

   # export TF_CPP_VMODULE="poplar_compiler=1,gpu_executable=5,nccl_all_reduce_thunk=5"
   # export TF_CPP_VMODULE="tx8_compiler=1"

   case $1 in
      0)
         echo "单机单卡 llava 7B TX8"
         # export GPU_NUM_DEVICES=4
         export CUDA_VISIBLE_DEVICES=0
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=4
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=1
         export PJRT_DEVICE=TX8
         # export TSM_DUMP_DATA=1
         # export TX8_MODEL_EXPORT_LEVEL=10
         # export TF_CPP_VMODULE="tx8_compiler=1"
         # export XLA_HLO_DEBUG=1
         python tx8/run_llava_train.py \
         --model_name_or_path /nvme4/lianghao/workspace/Llava1.5-7B/llava-15-7b-hf \
         --json_file_path /login_home/zhoujunjie/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
         --images_path /login_home/zhoujunjie/LLaVA-Pretrain \
         --num_train_epochs 100 \
         --per_device_train_batch_size 1 \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --cache_dir ./output/tmp \
         --block_size 4096 \
         --llama_num_hidden_layers 1 \
         --clip_num_hidden_layers 1 \
         --learning_rate 5e-5 \
         --optim adamw_torch \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --torch_dtype bfloat16 \
         ;;
      1)
         echo "单机单卡 llava 7B cuda"
         # export GPU_NUM_DEVICES=4
         export CUDA_VISIBLE_DEVICES=0
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=4
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=1
         export PJRT_DEVICE=TX8
         # export TSM_DUMP_DATA=1
         # export TX8_MODEL_EXPORT_LEVEL=10
         export XLA_HLO_DEBUG=1
         torchrun tx8/run_llava_train.py \
         --model_name_or_path /nvme4/lianghao/workspace/Llava1.5-7B/llava-15-7b-hf \
         --json_file_path /login_home/zhoujunjie/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
         --images_path /login_home/zhoujunjie/LLaVA-Pretrain \
         --num_train_epochs 100 \
         --per_device_train_batch_size 1 \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --cache_dir ./output/tmp \
         --llama_num_hidden_layers 1 \
         --clip_num_hidden_layers 1 \
         --block_size 128 \
         --optim adamw_torch \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --use_cuda \
         # --torch_dtype bfloat16 \
         ;;
      2)
         echo "单机单卡 llava 7B XLA GPU"
         # export GPU_NUM_DEVICES=4
         export CUDA_VISIBLE_DEVICES=0
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=4
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=1
         export PJRT_DEVICE=GPU
         # export TSM_DUMP_DATA=1
         # export TX8_MODEL_EXPORT_LEVEL=10
         #export XLA_HLO_DEBUG=1
         python tx8/run_llava_train.py \
         --model_name_or_path /nvme4/lianghao/workspace/Llava1.5-7B/llava-15-7b-hf \
         --json_file_path /login_home/zhoujunjie/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
         --images_path /login_home/zhoujunjie/LLaVA-Pretrain \
         --num_train_epochs 100 \
         --per_device_train_batch_size 1 \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --cache_dir ./output/tmp \
         --block_size 4096 \
         --llama_num_hidden_layers 1 \
         --clip_num_hidden_layers 1 \
         --optim adamw_torch \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --torch_dtype bfloat16 \
         ;;
      3) 
         echo "单机多卡 llava 7B TX8"
         # export GPU_NUM_DEVICES=4
         export CUDA_VISIBLE_DEVICES=0,1
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=4
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=2
         export PJRT_DEVICE=TX8
         export MASTER_ADDR=localhost
         export MASTER_PORT=12345
         # export TSM_DUMP_DATA=1
         # export TX8_MODEL_EXPORT_LEVEL=10
         # export XLA_HLO_DEBUG=1
         # export TF_CPP_VMODULE="tx8_compiler=1"
         python tx8/xla_spawn.py --num_cores 2 examples/pytorch/tx8/run_llava_train.py \
         --model_name_or_path /nvme4/lianghao/workspace/Llava1.5-7B/llava-15-7b-hf \
         --json_file_path /login_home/zhoujunjie/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
         --images_path /login_home/zhoujunjie/LLaVA-Pretrain \
         --num_train_epochs 100 \
         --per_device_train_batch_size 1 \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --cache_dir ./output/tmp \
         --block_size 128 \
         --llama_num_hidden_layers 1 \
         --clip_num_hidden_layers 1 \
         --optim adamw_torch \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "full_shard" \
         --fsdp_config tx8/fsdp_xla_config.json \
         --torch_dtype bfloat16 \
         # --mstt_config_name_or_path ./config_tensor.json
         # > llava_7b_mhlo_train_all.log
         ;;
      4)
         echo "单机单卡 llava 7B TX8"
         # export GPU_NUM_DEVICES=4
         export CUDA_VISIBLE_DEVICES=5
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=4
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=1
         export PJRT_DEVICE=TX8
         # export TSM_DUMP_DATA=1
         # export TX8_MODEL_EXPORT_LEVEL=10
         export TF_CPP_VMODULE="tx8_compiler=1"
         export XLA_HLO_DEBUG=1
         python tx8/run_llava_train.py \
         --model_name_or_path /login_home/zhoujunjie/tx8_train/Models/llava-hf/llava-15-7b-hf \
         --json_file_path /login_home/zhoujunjie/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
         --images_path /login_home/zhoujunjie/LLaVA-Pretrain \
         --num_train_epochs 100 \
         --per_device_train_batch_size 1 \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --llama_num_hidden_layers 1 \
         --clip_num_hidden_layers 1 \
         --cache_dir /login_home/zhoujunjie/tx8_train/output/tmp \
         --block_size 2048 \
         --optim adamw_torch \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --torch_dtype bfloat16 
         ;;
   esac

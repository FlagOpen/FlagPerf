# !/bin/sh
   clear
   export TF_CPP_MIN_LOG_LEVEL=0
   export TF_CPP_VMODULE="poplar_compiler=1"
   case $1 in
      0)
         export CUDA_VISIBLE_DEVICES=6
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=1
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=1
         export PJRT_DEVICE=TX8
         echo "单机单卡 Llama3-8B TX8_GPU"
         # /workspace/data/hugginface/models/Mixtral-8x7B-v0___1
         # /workspace/data/hugginface/models/llama3_model/Meta-Llama-3-70B
         #export XLA_HLO_DEBUG=1
         python tx8/run_xla_train.py \
         --model_name_or_path /login_home/chenjunhui/tx8/llama-3-8B/ \
         --dataset_name /login_home/yuancong/workplace/data/hugginface/datasets/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --do_train \
         --output_dir ./output/tmp/test-yc-clm \
         --overwrite_output_dir \
         --cache_dir ./output/tmp \
         --num_hidden_layers 1 \
         --block_size 8192 \
         --torch_dtype bfloat16 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5
         ;;
      1)
         export CUDA_VISIBLE_DEVICES=4,5,6,7
         export USE_TORCH_XLA=1
         #export PJRT_DEVICE=GPU
         export PJRT_DEVICE=TX8
	      export TX8_NUM_DEVICES=2
         export PJRT_LOCAL_WORLD_SIZE=2
         #export TSM_DUMP_DATA=1
         echo "单机多卡 llama3 8B TX8"
         # export XLA_HLO_DEBUG=1
         # --mstt_config_name_or_path ./config_tensor.json \
         # --resume_from_checkpoint /tmp/test-yc-clm/checkpoint-6 \
         python xla_spawn.py tx8/run_xla_train.py \
         --model_name_or_path /login_home/chenjunhui/tx8/llama-3-8B/ \
         --dataset_name /login_home/yuancong/workplace/data/hugginface/datasets/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir /tmp/test-yc-clm \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size 1024 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --logging_dir /tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "full_shard" \
         --fsdp_config ./fsdp_xla_config.json \
         ;;
   esac

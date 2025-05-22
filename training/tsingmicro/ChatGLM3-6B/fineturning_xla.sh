# !/bin/sh
   clear
   export TSM_ENABLE_COMPILE=0
   export TF_CPP_MIN_LOG_LEVEL=0
   export TF_CPP_VMODULE="tx8_compiler=1"
   export MASTER_ADDR="localhost"
   export MASTER_PORT="12355"
   case $1 in
      0)
         export CUDA_VISIBLE_DEVICES=5
         export USE_TORCH_XLA=1
         export TX8_NUM_DEVICES=1
         #export PJRT_DEVICE=GPU
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=1
         #export TSM_DUMP_DATA=1
         echo "单机单卡 ChatGLM3-6B TX8_GPU"
         # export XLA_HLO_DEBUG=0
         # export TSM_DUMP_DATA=0
         python tx8/run_xla_train.py \
         --model_name_or_path /login_home/chenjunhui/THUDM-chatglm3-6b \
         --dataset_name /login_home/chenjunhui/tx8/dataset/total_trainset_new.txt \
         --dataset_config_name default \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir /tmp/test-yc-clm \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size 4096 \
         --trust_remote_code True \
         --learning_rate 5e-5 \
         --save_strategy no \
         --optim adamw_torch \
         --use_flash_attn True \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --ddp_dp_sharding 1 \
         --megatron_tp_sharding 1\
         --torch_dtype bfloat16
         ;;
      1)
         export USE_TORCH_XLA=1
         export PJRT_DEVICE=TX8
         export CUDA_VISIBLE_DEVICES=0,1,2,3
         export PJRT_LOCAL_WORLD_SIZE=4
         export TX8_NUM_DEVICES=4
         # export GPU_NUM_DEVICES=4
         # export PJRT_DEVICE=GPU
         #export TSM_DUMP_DATA=0
         echo "单机多卡 ChatGLM3-6B TX8_GPU"
         python tx8/xla_spawn.py --ddp_dp_sharding 2 --megatron_tp_sharding 2 tx8/run_xla_train.py \
         --model_name_or_path /login_home/chenjunhui/THUDM-chatglm3-6b \
         --dataset_name /login_home/chenjunhui/tx8/dataset/total_trainset_new.txt \
         --dataset_config_name default \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir /tmp/test-yc-clm \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size 4096 \
         --trust_remote_code True \
         --learning_rate 5e-5 \
         --save_strategy no \
         --optim adamw_torch \
         --use_flash_attn True \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --torch_dtype bfloat16
         ;;
   esac

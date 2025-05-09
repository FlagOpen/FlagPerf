# !/bin/sh
   clear
   export TSM_ENABLE_COMPILE=0
   export TF_CPP_MIN_LOG_LEVEL=0
   # export TF_CPP_VMODULE="poplar_compiler=1"
   case $1 in
      10)
         export CUDA_VISIBLE_DEVICES=0,1,2,3
         # export CUDA_VISIBLE_DEVICES=4
         export USE_TORCH_XLA=1
         export PJRT_DEVICE=TX8
         export PJRT_GLOBAL_WORLD_SIZE = 2
         export PJRT_LOCAL_WORLD_SIZE=1
	 export TX8_NUM_DEVICES=1
         export PJRT_GROUP_RANK = 0
         # export PJRT_DEVICE=GPU
         echo "多机多卡 llama3 70B TX8"
         # python  examples/pytorch/tx8/run_xla_train.py \
         python xla_spawn.py run_xla_train.py \
         --model_name_or_path /data/chenkunfang/model_dir/Meta-Llama-3-70B \
         --dataset_name /data/yuancong/workplace/data/hugginface/datasets/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir /tmp/test-yc-clm \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size 4 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir /tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "full_shard" \
         --fsdp_config ./fsdp_xla_config.json \
         ;;
      esac

# !/bin/sh
   clear
   export XLA_THREAD_POOL_SIZE=4
   export XLA_IO_THREAD_POOL_SIZE=4
   export NPROC=4
   export OMP_NUM_THREADS=4
   export OMP_WAIT_POLICY=PASSIVE
   export TF_CPP_LOG_THREAD_ID=1
   export TX8_MAX_CONSTANT_SIZE=1
   #export TF_XLA_FLAGS="--xla_min_cluster_size=1--tf_xla_enable_xla_devices"
   export TF_CPP_MIN_LOG_LEVEL=0
   export TF_CPP_VMODULE="poplar_compiler=1"
   #export TF_CPP_VMODULE="poplar_compiler=1,tx8_executor=3,xla_graph_executor=1,init_python_bindings=1,tfrt_cpu_pjrt_client=1,tx8_threadpool=3,xla_device=1,tfrt_tx8_pjrt_client=3,pjrt_tx8_client=1,tx8_threadpool=3,hlo_pass_pipeline=5,hlo_constant_folding=5,tx8_hlo_constant_folding=5,hlo_evaluator=1,shape_util=1,hlo_evaluator=1"
   case $1 in
      0)
         echo "单机单卡 Baichuan2-13B GPU"
         export CUDA_VISIBLE_DEVICES=1 #指定哪些cuda卡
         export USE_TORCH_XLA=1        #动态图静态图
         export TX8_NUM_DEVICES=1      #全模型大 -》多卡
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=1
         # export TSM_DUMP_DATA=1      #覆盖  
         # export TX8_MODEL_EXPORT_LEVEL=11
         # export XLA_HLO_DEBUG=1
         # --mstt_config_name_or_path ./config_tensor.json \
         python tx8/run_xla_train.py \
         --model_name_or_path /login_home/zhangna/Baichuan2-13B/baichuan-inc--Baichuan2-13B-Base \
         --dataset_name /login_home/zhangna/dataset/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --trust_remote_code True \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --num_hidden_layers 1 \
         --block_size 2048 \
         --cache_dir ./output/tmp \
         --torch_dtype bfloat16 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         # --use_flash_attn True
         ;;

      1)
         echo "单机多卡 Baichuan2-13B GPU"
         export CUDA_VISIBLE_DEVICES=0,1,2,3 #指定哪些cuda卡
         export USE_TORCH_XLA=1        #动态图静态图
         export TX8_NUM_DEVICES=4      #全模型大 -》多卡
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=4
         # export TSM_DUMP_DATA=1      #覆盖  
         # export TX8_MODEL_EXPORT_LEVEL=10
         # export XLA_HLO_DEBUG=1
         export MASTER_ADDR=localhost
         export MASTER_PORT=12355
         python tx8/xla_spawn.py tx8/run_xla_train.py \
         --model_name_or_path /login_home/zhangna/Baichuan2-13B/baichuan-inc--Baichuan2-13B-Base \
         --dataset_name /login_home/zhangna/dataset/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --trust_remote_code True \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --num_hidden_layers 1 \
         --block_size 2048 \
         --cache_dir ./output/tmp \
         --torch_dtype bfloat16 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "full_shard" \
         --fsdp_config ./tx8/fsdp_xla_config.json \
         # --use_flash_attn True
         ;;

      2)
         echo "单机多卡 Baichuan2-13B GPU"
         export CUDA_VISIBLE_DEVICES=0,1,2,3 #指定哪些cuda卡
         export USE_TORCH_XLA=1        #动态图静态图
         export TX8_NUM_DEVICES=4    #全模型大 -》多卡
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=4
         export TSM_DUMP_DATA=1      #覆盖  
         export TX8_MODEL_EXPORT_LEVEL=10
         export MASTER_ADDR=localhost
         export MASTER_PORT=12355
         python tx8/xla_spawn.py tx8/run_xla_train.py \
         --model_name_or_path /login_home/zhangna/Baichuan2-13B/baichuan-inc--Baichuan2-13B-Base \
         --dataset_name /login_home/zhangna/dataset/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --trust_remote_code True \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --num_hidden_layers 8 \
         --block_size 2048 \
         --cache_dir ./output/tmp \
         --torch_dtype bfloat16 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "full_shard" \
         --fsdp_config ./tx8/fsdp_xla_config.json
         ;;
      
      3)
         echo "单机单卡 Baichuan2-13B cuda"
         export CUDA_VISIBLE_DEVICES=0 #指定哪些cuda卡
         export USE_TORCH_XLA=1        #动态图静态图
         export TX8_NUM_DEVICES=1      #全模型大 -》多卡
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=1
         # export TSM_DUMP_DATA=1      #覆盖  
         # export TX8_MODEL_EXPORT_LEVEL=11
         torchrun tx8/run_xla_train.py \
         --model_name_or_path /login_home/zhangna/Baichuan2-13B/baichuan-inc--Baichuan2-13B-Base \
         --dataset_name /login_home/zhangna/dataset/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --trust_remote_code True \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --num_hidden_layers 1 \
         --block_size 2048 \
         --cache_dir ./output/tmp \
         --torch_dtype bfloat16 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --use_cuda \
         ;;

   esac
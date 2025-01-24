# !/bin/sh
   clear
   export TSM_ENABLE_COMPILE=0
   export TF_CPP_MIN_LOG_LEVEL=0
   export TF_CPP_VMODULE="tx8_compiler=1"
   export MASTER_ADDR="localhost"
   export MASTER_PORT="12355"
   case $1 in
      0)
         export CUDA_VISIBLE_DEVICES=1
         export USE_TORCH_XLA=1
         export TX8_NUM_DEVICES=1
         #export PJRT_DEVICE=GPU
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=1
         #export TSM_DUMP_DATA=1
         echo "单机单卡 llama3 70B GPU"
         python run_xla_train.py \
         --fsdp_dp_sharding 1 \
         --megatron_tp_sharding 1 \
         --model_name_or_path /workplace/data/hugginface/models/llama3_model/Meta-Llama-3-70B \
         --dataset_name /workplace/data/hugginface/datasets/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir /tmp/test-yc-clm \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size 1024 \
         --learning_rate 5e-5 \
         --save_strategy no \
         --optim adamw_torch \
         --use_flash_attn True \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --torch_dtype bfloat16 \
         --logging_dir /tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         # --use-flagscale-dataset
         ;;
      1)
         export CUDA_VISIBLE_DEVICES=0,1
         export USE_TORCH_XLA=1
         #export PJRT_DEVICE=GPU
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=2
         #export TSM_DUMP_DATA=1
         echo "单机多卡 FSDP llama3 70B GPU"
         # export XLA_HLO_DEBUG=1
         # --mstt_config_name_or_path ./config_tensor.json \
         # --resume_from_checkpoint /tmp/test-yc-clm/checkpoint-6 \
         python xla_spawn.py --fsdp_dp_sharding 2 --megatron_tp_sharding 1 run_xla_train.py \
         --model_name_or_path /workplace/data/hugginface/models/llama3_model/Meta-Llama-3-70B \
         --dataset_name /workplace/data/hugginface/datasets/flagperf/wudao_llama3bpe_content_document \
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
      2)
         export CUDA_VISIBLE_DEVICES=0,1,2,3
         export USE_TORCH_XLA=1
         #export PJRT_DEVICE=GPU
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=4
         echo "单机多卡 FSDP+TP llama3 70B GPU"
         # export XLA_HLO_DEBUG=1
         # --mstt_config_name_or_path ./config_tensor.json \
         # --fsdp "full_shard" \ 
         python xla_spawn.py --fsdp_dp_sharding 2 --megatron_tp_sharding 2 run_xla_train.py \
         --model_name_or_path /workplace/data/hugginface/models/llama3_model/Meta-Llama-3-70B \
         --dataset_name /workplace/data/hugginface/datasets/flagperf/wudao_llama3bpe_content_document \
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
         --fsdp "hybrid_shard" \
         --fsdp_config ./fsdp_xla_config.json \
         ;;
      3)
         export CUDA_VISIBLE_DEVICES=0,1,2,3
         export USE_TORCH_XLA=1
         #export PJRT_DEVICE=GPU
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=4
         echo "单机多卡 llama3 70B cuda"
         # export XLA_HLO_DEBUG=1
         # --mstt_config_name_or_path ./config_tensor.json \
         torchrun --nnodes 1 --nproc_per_node 4 run_xla_train.py \
         --fsdp_dp_sharding 2 \
         --megatron_tp_sharding 2 \
         --model_name_or_path /nvme4/lianghao/workspace/Meta-Llama-3-70B \
         --dataset_name /workplace/data/hugginface/datasets/total_trainset_new.txt \
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
         --use_cuda \
         ;;
      4)
         #延时加载70B
         export CUDA_VISIBLE_DEVICES=0,1,2,3
         export USE_TORCH_XLA=1
         #export PJRT_DEVICE=GPU
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=4
         #export TSM_DUMP_DATA=1
         echo "单机多卡 llama3 70B GPU"
         # export XLA_HLO_DEBUG=1
         # --mstt_config_name_or_path ./config_tensor.json \
         # --resume_from_checkpoint /tmp/test-yc-clm/checkpoint-6 \ fsdp_tp_xla_config
         python xla_spawn.py --semaphore_number 1  --fsdp_dp_sharding 2 --megatron_tp_sharding 2 run_xla_train.py \
         --deferred_init_model_path /workplace/SPMD_TX8_DEVELOP/examples/llama3_70B_finetune/weights \
         --config_name /workplace/data/hugginface/models/llama3_model/Meta-Llama-3-70B \
         --tokenizer_name /workplace/data/hugginface/models/llama3_model/Meta-Llama-3-70B \
         --dataset_name /workplace/data/hugginface/datasets/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 10 \
         --torch_dtype bfloat16 \
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
         --fsdp "hybrid_shard" \
         --fsdp_config ./fsdp_xla_config.json \
         ;;
      esac

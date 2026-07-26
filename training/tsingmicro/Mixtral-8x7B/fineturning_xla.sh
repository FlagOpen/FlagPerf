# !/bin/sh
   clear
   export TF_CPP_MIN_LOG_LEVEL=0
   export MASTER_ADDR="localhost"
   export MASTER_PORT="12345"
   case $1 in
      0)
         export CUDA_VISIBLE_DEVICES=6
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=1
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=1
         export PJRT_DEVICE=TX8
         export TSM_DUMP_DATA=1
         export TX8_MODEL_EXPORT_LEVEL=11
         #export TF_CPP_VMODULE="tx8_compiler=1"
         echo "单机单卡 Mixtral-8x7B TX8_GPU"
         # export XLA_HLO_DEBUG=1
         python tx8/run_xla_train_fsdp_tp.py \
         --model_name_or_path /login_home/zhangxiaohe/data/huggingface/models/Mixtral-8x7B-v0___1 \
         --dataset_name /login_home/zhangxiaohe/data/huggingface/datasets/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --cache_dir ./output/tmp \
         --num_hidden_layers 1 \
         --block_size 4096 \
         --torch_dtype bfloat16 \
         --optim adafactor \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5
         ;;
      1)
         echo "train 单机单卡 Mitral-8x7B"
         # export GPU_NUM_DEVICES=4
         export CUDA_VISIBLE_DEVICES=6
         export USE_TORCH_XLA=1
         export TX8_NUM_DEVICES=1
         # export TX8_NUM_DEVICES=2  #多卡暂时由core dump错误
         export PJRT_DEVICE=TX8
         export TSM_DUMP_DATA=1
         export TX8_MODEL_EXPORT_LEVEL=11
         export TF_CPP_VMODULE="tx8_compiler=10"
         # export TSM_ENABLE_TX8_FUSION_PASS=0
         python tx8/run_xla_train.py \
         --model_name_or_path /nvme4/lianghao/workspace/Mixtral-8x7B/Mixtral-8x7B-v0___1 \
         --dataset_name /nvme4/lianghao/workspace/samsum \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --trust_remote_code True \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --num_hidden_layers 1 \
         --block_size 4096 \
         --cache_dir ./output/tmp \
         --torch_dtype bfloat16 \
         --optim adafactor \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         # --fsdp "full_shard" \
         # --fsdp_config ./tx8/fsdp_xla_config.json
         ;;
      2)
         echo "infer 单机单卡 Mitral-8x7B"
         # export GPU_NUM_DEVICES=4
         export CUDA_VISIBLE_DEVICES=3
         export USE_TORCH_XLA=1
         # export GPU_NUM_DEVICES=4
         # export PJRT_DEVICE=GPU
         export TX8_NUM_DEVICES=1
         export PJRT_DEVICE=TX8
         export TSM_DUMP_DATA=1
         export TX8_MODEL_EXPORT_LEVEL=1
         python tx8/test_infer.py
         ;;
      3)
         echo "train 2机2卡 Mitral-8x7B"
         export CUDA_VISIBLE_DEVICES=0,1,2,3
         export USE_TORCH_XLA=1
         export TX8_NUM_DEVICES=4 
         export PJRT_DEVICE=TX8
         export PJRT_LOCAL_WORLD_SIZE=4
         #export TSM_DUMP_DATA=1
         #export TX8_MODEL_EXPORT_LEVEL=11
         export TF_CPP_VMODULE="tx8_compiler=1"
         # export XLA_HLO_DEBUG=1
         export TSM_ENABLE_TX8_FUSION_PASS=0
         # --mstt_config_name_or_path tx8/config_tensor.json \
         # --use_dynamic_shape
         python tx8/xla_spawn.py --fsdp_dp_sharding 2 --megatron_tp_sharding 2 tx8/run_xla_train_fsdp_tp.py \
         --model_name_or_path /workplace/data/hugginface/models/Mixtral-8x7B-v0___1 \
         --dataset_name /workplace/data/hugginface/datasets/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --trust_remote_code True \
         --do_train \
         --output_dir ./output \
         --overwrite_output_dir \
         --num_hidden_layers 1 \
         --block_size 4096 \
         --torch_dtype bfloat16 \
         --cache_dir ./output/tmp \
         --overwrite_output_dir \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --logging_dir ./output/tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "hybrid_shard" \
         --fsdp_config ./tx8/fsdp_xla_config.json \
         ;;
   esac

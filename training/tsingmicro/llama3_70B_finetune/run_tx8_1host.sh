   clear
   #export HOST_LOG_LEVEL=0
   #export KCORE_LOG_LEVEL=0
   #export ENABLE_KCORE_CALC=0
   #export MEDIA_LOG_LEVEL=999
   export TX8_MODEL_EXPORT_LEVEL=5
   #export TSM_DUMP_DATA=1
   export TSM_ENABLE_COMPILE=1
   export TSM_NODES_SIZE_X_Y=1,1
   export CUDA_VISIBLE_DEVICES=1
   export MULTI_GRAPH_OPEN=0
   export MASTER_ADDR="127.0.0.1"
   export MASTER_PORT="12223"
   export USE_TORCH_XLA=1
   export PJRT_DEVICE=TX8
   export XLA_PARAMETER_WRAPPING_THREADSHOLD=1
   #export PJRT_GLOBAL_WORLD_SIZE=128
   export TF_CPP_MIN_LOG_LEVEL=0
   # export TF_CPP_VMODULE="poplar_compiler=1,tx8_executor=1"
   export TF_CPP_VMODULE="tx8_devices_manager=3,tx8_device_memory_allocator=3,tx8_threadpool=3,pjrt_computation_client=3,tx8_executor=3,poplar_compiler=1,tx8_execution_context=3,tx8_memory_pool=3,tx8_threadpool=3,tx8_utils=3,tx8_executable=3,pjrt_tx8_client=5,client=3,service=3"
   case $1 in
      1)
         export TSM_VISIBLE_DEVICES=27,1,1
         export PJRT_LOCAL_WORLD_SIZE=1
         #export TX8_COMPILER_CONFIG="--static_shape=1 --opt_search=1 --opt_group=opt_llama3_70b --opt_softmax=1--multi_graph=1"
         export TX8_COMPILER_CONFIG="--static_shape=1 --opt_search=1 --opt_group=opt_llama3_70b --opt_softmax=1"
         python run_xla_train.py \
         --fsdp_dp_sharding 1 \
         --megatron_tp_sharding 1 \
         --model_name_or_path /login_home/chenkunfang/model_dir/Meta-Llama-3-8B \
         --dataset_name /login_home/chenkunfang/model_dir/datasets/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir /login_home/chenkunfang/workspace/examples/llama3_70B_finetune/1_card_loss \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size 128 \
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
         --logging_steps 5  2>&1 | tee 1card_log.txt
         # --use-flagscale-dataset
         ;;
       2)
         export PJRT_DIST_SERVICE_ADDR=127.0.0.1:8099
         export TSM_VISIBLE_DEVICES=7,2,1
         export PJRT_LOCAL_WORLD_SIZE=2
         export TX8_COMPILER_CONFIG="--static_shape=1 --opt_search=1 --opt_group=opt_llama3_70b --opt_softmax=1--multi_graph=1--chip_num=2,1"
         python xla_spawn.py --fsdp_dp_sharding 2 --megatron_tp_sharding 1 run_xla_train.py \
         --model_name_or_path /login_home/chenkunfang/model_dir/Meta-Llama-3-8B \
         --dataset_name /login_home/chenkunfang/model_dir/datasets/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir /login_home/chenkunfang/workspace/examples/llama3_70B_finetune/2_cards_loss \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size 1024 \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --logging_dir /tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "full_shard" \
         --gradient_accumulation_steps 2 \
         --fsdp_config ./fsdp_xla_config.json  2>&1 | tee 2cards_log.txt
         ;;
         esac

   clear
   # export HOST_LOG_LEVEL=0
   #export KCORE_LOG_LEVEL=0
   #export ENABLE_KCORE_CALC=0
   #export MEDIA_LOG_LEVEL=999
   export TX8_MODEL_EXPORT_LEVEL=5
   # export TSM_DUMP_DATA=1
   #export TSM_ENABLE_COMPILE=0
   #export MULTI_GRAPH_OPEN=1
   export CUDA_VISIBLE_DEVICES=1
   export PJRT_DIST_SERVICE_ADDR=10.152.21.13:8099
   export MASTER_ADDR="10.152.21.13"
   export TSM_NODES_SIZE_X_Y=2,2
   export TSM_VISIBLE_DEVICES=0,4,8
   export USE_TORCH_XLA=1
   export PJRT_DEVICE=TX8
   export PJRT_GLOBAL_WORLD_SIZE=128
   export PJRT_LOCAL_WORLD_SIZE=32
   export TX8_NUM_DEVICES=32
  #  export TX8_COMPILER_CONFIG="--txccl_mode=8--txccl_core=4--static_shape=1 --opt_search=1 --opt_group=opt_llama3_70b --opt_softmax=1 --chip_num=8,16--multi_graph=1"
   export TX8_COMPILER_CONFIG="--txccl_mode=8--txccl_core=4--static_shape=1 --opt_search=1 --opt_group=opt_llama3_70b --opt_softmax=1 --chip_num=8,16"
   export TF_CPP_MIN_LOG_LEVEL=0
   # export TF_CPP_VMODULE="poplar_compiler=1,tx8_executor=1"
   # export TF_CPP_VMODULE="tx8_devices_manager=3,tx8_device_memory_allocator=3,tx8_threadpool=3,pjrt_computation_client=3,tx8_executor=3,poplar_compiler=1,tx8_execution_context=3,tx8_memory_pool=3,tx8_threadpool=3,tx8_utils=3,tx8_executable=3,pjrt_tx8_client=5,client=10,service=10"
   export TF_CPP_VMODULE="tx8_devices_manager=3,tx8_device_memory_allocator=3,tx8_threadpool=3,pjrt_computation_client=3,tx8_executor=3,poplar_compiler=1,tx8_execution_context=3,tx8_memory_pool=3,tx8_threadpool=3,tx8_utils=3,tx8_executable=3,pjrt_tx8_client=5,client=3,service=3"
   case $1 in
      1)
         export PJRT_GROUP_RANK=0
         ;;
      2)
         export PJRT_GROUP_RANK=1
         ;;
      3)
         export PJRT_GROUP_RANK=2
         ;;
      4)
         export PJRT_GROUP_RANK=3
         ;;
      esac
   echo "多机多卡 llama3 70B TX8"
   python xla_spawn.py --fsdp_dp_sharding 128 --megatron_tp_sharding 1 run_xla_train.py \
         --model_name_or_path /login_home/chenkunfang/model_dir/Meta-Llama-3-70B \
         --dataset_name /login_home/chenkunfang/model_dir/datasets/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --do_train \
         --output_dir  /login_home/chenkunfang/workspace/examples/llama3_70B_finetune/loss \
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
         --fsdp_config ./fsdp_xla_config.json  2>&1 | tee host$1.txt

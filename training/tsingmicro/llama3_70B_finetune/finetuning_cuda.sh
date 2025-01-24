# !/bin/sh
   # clear
   export TSM_ENABLE_COMPILE=0
   export TF_CPP_MIN_LOG_LEVEL=0
   export TF_CPP_VMODULE="poplar_compiler=1"
   echo '$1:' 'tp_num:' $1
   echo '$2:' 'dp_num:' $2
   echo '$3:' 'ga_num:' $3
   echo '$4:' 'nnodes:' $4
   echo '$5:' 'node_rank:' $5
   echo '$6:' 'nproc_per_node:' $6
   echo '$7:' 'block_size:' $7


   device_num=$(($((1))*$((2))))
   master_addr='127.0.0.1'
   master_port=28356

   case $1$2$3 in
      *)
         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
         # export CUDA_VISIBLE_DEVICES=4,5,6,7
         export PJRT_DEVICE=GPU
         # export FSDP_CPU_RAM_EFFICIENT_LOADING=1
         # export ACCELERATE_USE_FSDP=1
         export PJRT_LOCAL_WORLD_SIZE=$6
         echo "单机多卡 llama3 70B GPU"
         ib1_str=`echo $(ibstatus | grep Infiniband | grep -v bond | awk '{print $3}')`
         ib1_str=`echo ${ib1_str} |tr -d \' `
         IFS=' ' read -r -a ib1_tuple <<< "$ib1_str"
         ib1=${ib1_tuple[-2]}
         ib2=${ib1_tuple[-1]}
         echo ${ib1}:1,${ib2}:1
         # /data/LM/hf/Meta-Llama-3-70B/
         # /data/LM/hf/Meta-Llama-3-8B
         # /data/LM/hf/Llama-3-70B-zhiyuan_ckpt_start-to-hf
         # /data/flagperf/wudao_llama3bpe_content_document
         # /data/zhiyuan/SAMPLE50B_llama3/llama3_dataset/dedup-md5-pile-pile-cc_text_document
         # --use-flagscale-dataset \
         torchrun --nnodes ${4} --node_rank ${5} --master_addr ${master_addr} --master_port ${master_port} --nproc_per_node ${6} run_xla_train.py \
         --model_name_or_path /data/LM/hf/Meta-Llama-3-70B/ \
         --dataset_name /data/flagperf/wudao_llama3bpe_content_document \
         --dataset_config_name default \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --num_hidden_layers 1 \
         --gradient_accumulation_steps ${3} \
         --do_train \
         --output_dir ./output_cuda_${1}${2}${3} \
         --overwrite_output_dir \
         --cache_dir /tmp \
         --block_size ${7} \
         --optim adamw_torch \
         --learning_rate 5e-5 \
         --save_strategy no \
         --weight_decay 0.0 \
         --adam_beta2 0.999 \
         --lr_scheduler_type "cosine" \
         --warmup_ratio 0.0 \
         --fsdp_dp_sharding ${2} \
         --megatron_tp_sharding ${1} \
         --logging_dir /tmp/logs \
         --logging_strategy steps \
         --logging_steps 5 \
         --fsdp "full_shard" \
         --fsdp_config ./fsdp_xla_config.json \
         --low_cpu_mem_usage false \
         --use_cuda \
         --use_flash_attn \
         --torch_dtype bfloat16 \
         # --max_steps 5 \
         # --tsprobe_config /xxx/ts_accuracy_tools/tsprobe/config/config.json \
         # --dump_path /xxx/dump_cuda_${1}${2}${3} \
         ;;

      esac


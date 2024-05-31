#!/bin/bash
DS_CONFIG=$1
DATA_DIR=$2
MODEL_NAME_OR_PATH=$DATA_DIR/LLaVA-Pretrain/checkpoints/vicuna-13b-v1.5
DATA_PATH=$DATA_DIR/LLaVA-Finetune/llava_v1_5_mix665k.json
IMAGE_FOLDER=$DATA_DIR/LLaVA-Finetune/data
VISION_TOWER=$DATA_DIR/LLaVA-Pretrain/checkpoints/openai/clip-vit-large-patch14-336
OUTPUT_DIR=$DATA_DIR/Output/checkpoints_finetune/llava-v1.5-13b
PRETRAIN_MM_MLP_ADAPTER=$DATA_DIR/Output/checkpoints_pretrain/llava-v1.5-13b/mm_projector.bin
MBS=$3
GAS=$4
deepspeed train/train_mem.py \
    --deepspeed $DS_CONFIG \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_TOWER \
    --pretrain_mm_mlp_adapter $PRETRAIN_MM_MLP_ADAPTER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $MBS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GAS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
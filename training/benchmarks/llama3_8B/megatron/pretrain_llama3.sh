#!/bin/bash

# Runs the "LLaMA3-8B" parameter model
export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1

ls /workspace
# algorithm args

MODEL_ARGS=" \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1.0e-5 \
    --global-batch-size 512 \
    --attention-softmax-in-fp32"

OPT_ARGS=" \
    --lr 1.0e-5 \
    --train-iters 300 \
    --lr-decay-iters 300 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.008"
    
ALGO_ARGS="$MODEL_ARGS $OPT_ARGS"

# data args

DATA_ARGS=" \
    --data-path /data/llama3_8b_pretrain/wudao_llama3bpe_content_document \
    --tokenizer-type Llama3Tokenizer \
    --split 100,0,0
"

# training args

TRAINING_ARGS=" \
    --micro-batch-size 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --bf16
"

# vendor args

VENDOR_ARGS=" \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer \
    --use-mcore-models \
    --use-flash-attn
"

OUTPUT_ARGS=" --log-interval 1"

run_cmd="torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 10.1.2.59 --master_port 29501 /workspace/Megatron-LM/pretrain_gpt.py \
    $ALGO_ARGS \
    $DATA_ARGS \
    $TRAINING_ARGS \
    $VENDOR_ARGS \
    $OUTPUT_ARGS"

echo ${run_cmd}
eval ${run_cmd}

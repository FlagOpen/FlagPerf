#!/bin/bash
# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# args

DATA_DIR=$1
GPUS_PER_NODE=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
MASTER_PORT=$6
MEGAPATH=$7
MBS=$8
ITERS=$9
TP=${10}
PP=${11}
TDIR=${12}
ADAPT=${13}

# Runs the "LLaMA3-70B" parameter model
export PYTHONPATH=$PYTHONPATH:$MEGAPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# algorithm args

MODEL_ARGS=" \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
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
    --train-iters $ITERS \
    --lr-decay-iters $ITERS \
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
    --data-path $DATA_DIR/wudao_llama3bpe_content_document \
    --tokenizer-type Llama3Tokenizer \
    --tokenizer-model $TDIR \
    --split 100,0,0
"

# training args

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
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

source $ADAPT
run_cmd="torchrun $DISTRIBUTED_ARGS $MEGAPATH/pretrain_gpt.py \
    $ALGO_ARGS \
    $DATA_ARGS \
    $TRAINING_ARGS \
    $VENDOR_ARGS \
    $OUTPUT_ARGS"

echo ${run_cmd}
eval ${run_cmd}

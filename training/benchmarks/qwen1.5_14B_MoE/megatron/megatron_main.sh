#!/bin/bash
# Runs Qwen1.5 14B(8*1.8B) MoE model on 8xA800 GPUs

set -e
DATA_PATH=$1
DATASET_PATH="${DATA_PATH}llama_00_text_document"
TOKENIZER_PATH="${DATA_PATH}tokenizer"

GPUS_PER_NODE=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
MASTER_PORT=$6

BATCH_SIZE=$9
GLOBAL_BATCH_SIZE=$10
SEQ_LEN=$11
PAD_LEN=$12
PR=$13
TP=$14
PP=$15

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

if [ $PR = fp16 ]; then
    PR_ARGS=" \
		--fp16"
elif [ $PR = bf16 ]; then
    PR_ARGS=" \
        --bf16"
fi

MODEL_ARGS=" \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --ffn-hidden-size 5504 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --use-rotary-position-embeddings \
    --no-rope-fusion \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --add-qkv-bias \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --apply-query-key-layer-scaling \
    --rotary-seq-len-interpolation-factor 1 \
    --seq-length ${SEQ_LEN} \
    --no-load-optim \
    --no-load-rng \
    --max-position-embeddings ${PAD_LEN} \

"

MOE_ARGS=" \
    --moe-router-topk 1 \
    --num-experts 8 \
    --moe-aux-loss-coeff 1e-2 \
    --expert-model-parallel-size 1 \
    --moe-router-load-balancing-type aux_loss
"

DATA_ARGS=" \
    --data-path ${DATASET_PATH} \
    --patch-tokenizer-type Qwen2Tokenizer \
    --load ${TOKENIZER_PATH} \
    --dataset LLama-Pretrain-Idxmap \
    --split 99,1,0 \
    --num-workers 8 \
    --extra-vocab-size 293 \
"

LOGGING_ARGS=" \
    --log-interval 1 \
"

TRAINING_ARGS=" \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --init-method-std 0.008 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --use-flash-attn \
    --micro-batch-size ${BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters 20000 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --transformer-impl transformer_engine \
    --recompute-activations
"

EVAL_ARGS=" \
    --eval-interval 10000 \
    --eval-iters 10 \

"

MODEL_PARALLEL_ARGS=" \
    --use-distributed-optimizer \
    --sequence-parallel \
"

run_cmd="torchrun ${DISTRIBUTED_ARGS} ${EXECPATH}/pretrain_mcore_qwen.py 
    ${MODEL_ARGS} ${PR_ARGS} ${MOE_ARGS} ${DATA_ARGS} ${LOGGING_ARGS} ${TRAINING_ARGS} ${EVAL_ARGS} ${MODEL_PARALLEL_ARGS}"

echo ${run_cmd}
eval ${run_cmd}
set +x

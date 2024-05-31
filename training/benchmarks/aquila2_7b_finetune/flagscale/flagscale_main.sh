#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/raid/dataset/aquila2_7b_finetune/FlagScale
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO

DATA_DIR=$1
WORLD_SIZE=$2
TRAIN_SAMPLES=$3
TP=$4
PP=$5
M_BATCHSIZE=$6
G_BATCHSIZE=$7
SEQLENGTH=$8
FLASH_ATTN=$9

echo $DATA_DIR
echo $WORLD_SIZE
echo $TRAIN_SAMPLES
echo $TP
echo $PP
echo $M_BATCHSIZE
echo $G_BATCHSIZE
echo $SEQLENGTH
echo $FLASH_ATTN

TRAIN_DATA_PATH=$DATA_DIR/dataset/alpaca_data_train.jsonl
VOCAB_FILE=$DATA_DIR/tokenizer/vocab.json
MERGE_FILE=$DATA_DIR/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$DATA_DIR/tokenizer/special_tokens.txt
LOAD_CHECKPOINT_PATH=$DATA_DIR/checkpoints/

DISTRIBUTED_ARGS="
    --nproc_per_node $WORLD_SIZE \

"

if [ "$FLASH_ATTN" = "True" ]; then
    TRAINING_ARGS="
        --train-samples $TRAIN_SAMPLES \
        --dataloader-type cyclic \
        --eval-iters 0 \
        --eval-interval 20 \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --make-vocab-size-divisible-by 64 \
        --micro-batch-size $M_BATCHSIZE \
        --global-batch-size $G_BATCHSIZE \
        --disable-bias-linear \
        --use-distributed-optimizer \
        --use-flash-attn
    "
else
    TRAINING_ARGS="
        --train-samples $TRAIN_SAMPLES \
        --dataloader-type cyclic \
        --eval-iters 0 \
        --eval-interval 20 \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --make-vocab-size-divisible-by 64 \
        --micro-batch-size $M_BATCHSIZE \
        --global-batch-size $G_BATCHSIZE \
        --disable-bias-linear \
        --use-distributed-optimizer
    "
fi

MIXED_PRECISION_ARGS="
    --bf16
"

DATA_ARGS="
    --train-data-path $TRAIN_DATA_PATH \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008 \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --merge-file $MERGE_FILE
"

NETWORK_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length $SEQLENGTH \
    --max-position-embeddings $SEQLENGTH \
    --layernorm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-in-fp32 \
    --no-position-embedding \
    --swiglu \
    --multiple-of 256 \
    --apply-layernorm-rms \
    --rotary-interleaved-patch \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 42
"

REGULARIZATION_ARGS="
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0
"

LEARNING_RATE_ARGS="
    --lr 9.65e-6 \
    --lr-decay-style linear \
    --lr-warmup-fraction 0.1 \
    --min-lr 0.0
"

CHECKPOINTING_ARGS="
    --load $LOAD_CHECKPOINT_PATH
    --no-load-optim \
    --no-load-rng \
    --finetune
"
LOGGING_ARGS="
    --log-interval 1 \
"


cmd="torchrun $DISTRIBUTED_ARGS finetune_aquila.py \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $LOGGING_ARGS
    "
echo $cmd
eval $cmd



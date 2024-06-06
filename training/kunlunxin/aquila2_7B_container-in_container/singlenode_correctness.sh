#!/bin/bash
SCALEHOME=/mnt/FlagScale

LOAD_CKPT_PATH=/mnt/ckpt/aquila/7B/load_aiplat
TB_PATH=/mnt/tb/aquila7B
mkdir $TB_PATH

DATADIR=/mnt/wudao_pretrain
DATASET=wudao_pretrain_text_document
TRAININGSAMPLES=80000

TP=4
PP=2
MBS=1
GBS=16

NODERANK=0
NNODES=1
MASTERADDR=102.168.1.2
MASTERPORT=29501

export PYTHONPATH=$PYTHONPATH:$SCALEHOME
export XDNN_FC_GEMM_DTYPE="float16"

VOCAB_FILE=$SCALEHOME/examples/aquila/tokenizer/vocab.json
MERGE_FILE=$SCALEHOME/examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$SCALEHOME/examples/aquila/tokenizer/special_tokens.txt
TRAINING_ITERS=$(expr $TRAININGSAMPLES / $GBS)

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes $NNODES \
    --node_rank $NODERANK \
    --master_addr $MASTERADDR \
    --master_port $MASTERPORT
"

TRAINING_ARGS="
    --train-iters $TRAINING_ITERS \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --sequence-parallel \
    --use-flash-attn
"

MIXED_PRECISION_ARGS="
    --fp16 \
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
"

DATA_ARGS="
    --data-path $DATADIR/$DATASET \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --data-impl mmap \
    --split 1
"

NETWORK_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --layernorm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-in-fp32 \
    --no-position-embedding \
    --swiglu \
    --multiple-of 256 \
    --apply-layernorm-rms \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 1234
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
    --lr 0.0003 \
    --min-lr 1.0e-5 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --lr-warmup-fraction .01 \
"

CKPT_ARGS="--load $LOAD_CKPT_PATH --no-load-optim --no-load-rng --finetune"

LOG_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TB_PATH \
    --tensorboard-log-interval 1
"

cmd="torchrun $DISTRIBUTED_ARGS $SCALEHOME/pretrain_gpt.py \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $CKPT_ARGS \
              $LOG_ARGS \
              $LEARNING_RATE_ARGS
    "
echo $cmd
eval $cmd


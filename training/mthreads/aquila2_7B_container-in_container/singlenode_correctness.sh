#!/bin/bash
SCALEHOME=/home/dist/zhiyuan-test/FlagScale
export OMP_NUM_THREADS=4
export MUSA_KERNEL_TIMEOUT=3600000
export CUDA_DEVICE_MAX_CONNECTIONS=1

LOAD_CKPT_PATH=/home/dist/zhiyuan/checkpoint/multinode/load_aiplat/
TB_PATH=/home/aquila7B/$(date +%m%d-%H%M%S)
mkdir $TB_PATH

DATADIR=/home/dist/zhiyuan/wudao_pretrain
DATASET=wudao_pretrain_text_document
TRAININGSAMPLES=80000

TP=2
PP=2
MBS=1
GBS=16

NODERANK=0
NNODES=1
MASTERADDR=10.74.47.78
MASTERPORT=29501

export PYTHONPATH=$PYTHONPATH:$SCALEHOME

VOCAB_FILE=$SCALEHOME/aquila/tokenizer/vocab.json
MERGE_FILE=$SCALEHOME/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$SCALEHOME/aquila/tokenizer/special_tokens.txt

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes $NNODES \
    --node_rank $NODERANK \
    --master_addr $MASTERADDR \
    --master_port $MASTERPORT
"

TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --use-flash-attn \
    --no-gradient-accumulation-fusion \
    --distributed-backend mccl \
    --device-type mthreads
"

MIXED_PRECISION_ARGS="
    --fp16 \
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
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
    --rotary-interleaved-patch \
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
    --lr 2.0e-5 \
    --min-lr 2.0e-6 \
    --lr-decay-style cosine \
    --lr-warmup-samples 8
"

CKPT_ARGS="--load $LOAD_CKPT_PATH --no-load-optim --no-load-rng --finetune"

LOG_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TB_PATH \
    --tensorboard-log-interval 1
"

cmd="torchrun $DISTRIBUTED_ARGS $SCALEHOME/megatron/pretrain_gpt.py \
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

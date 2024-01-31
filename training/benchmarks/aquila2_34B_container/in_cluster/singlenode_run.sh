#!/bin/bash

ifconfig

git log | head -n 5
lrank=$OMPI_COMM_WORLD_LOCAL_RANK
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

export NCCL_IB_TIMEOUT=22
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

SCALEHOME=$1

DATADIR=$2
DATASET=$3
TRAININGSAMPLES=$4

TP=$5
PP=$6
MBS=$7
GBS=$8

MASTER_ADDR=$9
MASTER_PORT=${10}
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT


export PYTHONPATH=$PYTHONPATH:$SCALEHOME/megatron

VOCAB_FILE=$SCALEHOME/aquila/tokenizer/vocab.json
MERGE_FILE=$SCALEHOME/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$SCALEHOME/aquila/tokenizer/special_tokens.txt


TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
    --sequence-parallel 
" 

MIXED_PRECISION_ARGS="
    --fp16 \
    --embedding-weights-in-fp32 \
    --rotary-position-embeddings-in-fp32 \
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
    --num-layers 60 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --hidden-dim-multiplier 1.3 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --layernorm-epsilon 1e-5 \
    --layernorm-init-weight 0.3 \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-in-fp32 \
    --no-position-embedding \
    --swiglu \
    --multiple-of 4096 \
    --apply-layernorm-rms \
    --untie-embeddings-and-output-weights \
    --no-gradient-accumulation-fusion
"

INITIALIZATION_ARGS="
    --init-method-std 0.0165 \
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
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --lr-warmup-samples 8
"

LOG_ARGS="
    --log-interval 1 \
"

cd $SCALEHOME
APP="python3 -u ./megatron/pretrain_gpt.py \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $LOG_ARGS 
    "
case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac

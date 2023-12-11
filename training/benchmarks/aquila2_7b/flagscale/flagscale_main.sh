export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale
export CUDA_DEVICE_MAX_CONNECTIONS=1

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

DATA_PATH=$DATA_DIR/pile_wikipedia_demo
VOCAB_FILE=$DATA_DIR/tokenizer/vocab.json
MERGE_FILE=$DATA_DIR/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$DATA_DIR/tokenizer/special_tokens.txt

DISTRIBUTED_ARGS="
    --nproc_per_node $WORLD_SIZE \
"

if [ "$FLASH_ATTN" = "True" ]; then
    TRAINING_ARGS="
        --train-samples $TRAIN_SAMPLES \
        --eval-iters 0 \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --micro-batch-size $M_BATCHSIZE \
        --global-batch-size $G_BATCHSIZE \
        --disable-bias-linear \
        --use-distributed-optimizer \
        --use-flash-attn
    "
else
    TRAINING_ARGS="
        --train-samples $TRAIN_SAMPLES \
        --eval-iters 0 \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
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
    --data-path $DATA_PATH \
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
    --seq-length $SEQLENGTH \
    --max-position-embeddings $SEQLENGTH \
    --layernorm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
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
    --lr-decay-style cosine 
"

cmd="torchrun $DISTRIBUTED_ARGS pretrain_aquila.py \
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



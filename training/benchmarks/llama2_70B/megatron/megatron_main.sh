export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATA_DIR=$1
GPUS_PER_NODE=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
MASTER_PORT=$6
TRAIN_SAMPLES=$7
TP=$8
PP=$9
M_BATCHSIZE=${10}
G_BATCHSIZE=${11}
SEQLENGTH=${12}
FLASH_ATTN=${13}
RECOMPUTE=${14}
VENDOR_SHELL=${15}

echo $DATA_DIR
echo $GPUS_PER_NODE
echo $NNODES
echo $NODE_RANK
echo $MASTER_ADDR
echo $MASTER_PORT
echo $TRAIN_SAMPLES
echo $TP
echo $PP
echo $M_BATCHSIZE
echo $G_BATCHSIZE
echo $SEQLENGTH
echo $FLASH_ATTN
echo $RECOMPUTE
echo $VENDOR_SHELL

DATA_PATH=$DATA_DIR/llama_00_text_document
TOKENIZER_PATH=$DATA_DIR/tokenizer/tokenizer.model

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
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
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 1
"

NETWORK_ARGS="
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --ffn-hidden-size 28672 \
    --seq-length $SEQLENGTH \
    --max-position-embeddings $SEQLENGTH \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 4096 \
    --sequence-parallel \
    --untie-embeddings-and-output-weights
"

if [ "$RECOMPUTE" = "True" ]; then
    RECOMPUTE_ARGS="
        --recompute-activations
    "
else
    RECOMPUTE_ARGS=""
fi

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 1234 
"

REGULARIZATION_ARGS="
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0
"

LEARNING_RATE_ARGS="
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction .01 
"

LOGGING_ARGS="
    --log-interval 1
"

CODE_PATH="/workspace/FlagScale/pretrain_llama.py"

source $VENDOR_SHELL
cmd="torchrun $DISTRIBUTED_ARGS $CODE_PATH \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $RECOMPUTE_ARGS \
              $LOGGING_ARGS
    "
echo $cmd
eval $cmd

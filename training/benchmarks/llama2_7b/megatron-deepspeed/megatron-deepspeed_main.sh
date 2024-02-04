export MEGFATRON_DS_HOME="/workspace/Megatron-DeepSpeed"

DATA_DIR=$1
GPUS_PER_NODE=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
MASTER_PORT=$6
TP=$7
PP=$8
MICRO_BATCH_SIZE=$9
GLOBAL_BATCH_SIZE=${10}
ACCUMULATE_STEPS=${11}
SEQ_LENGTH=${12}
TRAIN_STEPS=${13}

echo $DATA_DIR
echo $GPUS_PER_NODE
echo $NNODES
echo $NODE_RANK
echo $MASTER_ADDR
echo $MASTER_PORT
echo $TP
echo $PP
echo $MICRO_BATCH_SIZE
echo $GLOBAL_BATCH_SIZE
echo $ACCUMULATE_STEPS
echo $SEQ_LENGTH
echo $TRAIN_STEPS

cp ./tokenizer.py ${MEGFATRON_DS_HOME}/megatron/tokenizer/tokenizer.py
BASE_PATH=`pwd`
DS_CONFIG=${BASE_PATH}/ds_config.json

DATASET=$DATA_DIR/RedPajama-Data-1T-Sample_text_document
TOKENIZER_PATH=${BASE_PATH}/preprocess/tokenizer.json

ZERO_STAGE=1

HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
NUM_LAYERS=32 # e.g. llama-13b: 40
NUM_HEADS=32 # e.g. llama-13b: 40
NUM_KV_HEADS=32 # llama2 70B uses GQA

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=20 #200
WEIGHT_DECAY=0.1
GRAD_CLIP=1

USE_DEEPSPEED=True
######################################

cat <<EOT > $DS_CONFIG
{
  "deepspeed": true,
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "gradient_accumulation_steps": $ACCUMULATE_STEPS,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "cpu_offload": false,
    "contiguous_gradients": false,
    "overlap_comm": false,
    "reduce_bucket_size": 5000000,
    "allgather_bucket_size": 5000000
  },
  "bf16": {
    "enabled": true
  },
  "data_types": {
    "grad_accum_dtype": "bf16"
  }
}
EOT

#

ds_args=""
ds_args=" --deepspeed=$USE_DEEPSPEED ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_port $MASTER_PORT"

cmd="torchrun $DISTRIBUTED_ARGS \
       /workspace/Megatron-DeepSpeed/pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type HFTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --eval-iters 0 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --deepspeed \
       --deepspeed_config=$DS_CONFIG \
       --no-pipeline-parallel \
       --use-flash-attn-v2 \
       #--recompute-activations \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       $ds_args
    "
echo $cmd
eval $cmd
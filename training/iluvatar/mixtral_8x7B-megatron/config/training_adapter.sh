echo "[Prompt] iluvatar adaption is not NULL, for other Vendors"
GPUS_PER_NODE=$2
NNODES=$3
NODE_RANK=$4
MEGAPATH=$7
MBS=$8
ITERS=$9
TP=${10}
PP=${11}
MASTERADDR=10.31.10.149
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH=$MEGAPATH/megatron:$MEGAPATH/megatron:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

##非四机暂设为8，四机16卡num-layers=32
MODEL_ARGS=" \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --max-position-embeddings 32768 \
    --seq-length 2048 \
    --swiglu \
    --normalization RMSNorm \
    --global-batch-size 128 \
    --disable-bias-linear \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-base 1000000.0"

OPT_ARGS=" \
    --lr 1.0e-5 \
    --min-lr 1e-05 \
    --train-iters $ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --init-method-std 0.02"

MOE_ARGS=" \
    --moe-router-topk 2 \
    --num-experts 8 \
    --moe-aux-loss-coeff 1e-2 \
    --expert-model-parallel-size 4 \
    --moe-router-load-balancing-type aux_loss \
    "
TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --bf16
"
DATA_ARGS=" \
    --data-path $MEGAPATH/data_dir/pile_wikipedia_demo \
    --tokenizer-type QwenTokenizer \
    --split 99,1,0
"
# vendor args
VENDOR_ARGS=" \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer \
    --use-mcore-models 
"
DISTRIBUTED_ARGS="
    --rdzv_id default \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master-addr $MASTERADDR \
    --master-port 53497 \
    --redirects 3 \
    --tee 3
"
setup_args="
    --tensorboard-log-interval 1 \
    --wandb-project mixtral \
    --wandb-exp-name mixtral-8x7b \
    --save-interval 10000 \
    --save  $MEGAPATH/outputs/checkpoints \
    --norm-epsilon 1e-05 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --lr-warmup-iters 500 \
    --tokenizer-path $MEGAPATH/data_dir/Qwen1___5-7B-Chat-GPTQ-Int8/ \
    --vocab-file $MEGAPATH/examples/aquila/tokenizer/vocab.json \
    --merge-file $MEGAPATH/examples/aquila/tokenizer/merges.txt \
    --special-tokens-file $MEGAPATH/examples/aquila/tokenizer/special_tokens.txt \
    --vocab-size 151851 \
    --make-vocab-size-divisible-by 64 \
    --wandb-save-dir $MEGAPATH/outputs/wandb \
    --tensorboard-dir $MEGAPATH/outputs/tensorboard \
    --load $MEGAPATH/outputs/checkpoints
"

ALGO_ARGS="$MODEL_ARGS $OPT_ARGS $GQA_ARGS $MOE_ARGS $setup_args"
##--log_dir $MEGAPATH/outputs/logs/details/host/20240627_060549.688979

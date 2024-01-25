VOCAB_FILE=$SCALEHOME/examples/aquila/tokenizer/vocab.json
MERGE_FILE=$SCALEHOME/examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$SCALEHOME/examples/aquila/tokenizer/special_tokens.txt

TB_PATH=./aquila7B_perfermance
mkdir -p $TB_PATH

DISTRIBUTED_ARGS="
    --nproc_per_node 16 \
    --nnodes $NNODES \
    --node_rank $NODERANK \
    --master_addr $MASTERADDR \
    --master_port $MASTERPORT
"

TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --eval-interval 2000 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --num-layers-per-virtual-pipeline-stage 8 \
    --use-flash-attn \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-method-per-stage 3 0 1 1 \
    --recompute-num-layers-per-stage 3 1 1 0 \
    --sequence-parallel \
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --attention-softmax-in-fp32 \
    --embedding-weights-in-fp32 \
    --rotary-position-embeddings-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --data-path $DATADIR/$DATASET \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
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

LOG_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TB_PATH \
    --tensorboard-log-interval 1
"
# for iluvatar performance testing, we need change corresponding singlenode_run.sh :
# cmd="torchrun $DISTRIBUTED_ARGS $SCALEHOME/megatron/pretrain_gpt.py \
#               $TRAINING_ARGS \
#               $MIXED_PRECISION_ARGS \
#               $DATA_ARGS \
#               $NETWORK_ARGS \
#               $INITIALIZATION_ARGS \
#               $REGULARIZATION_ARGS \
#               $LEARNING_RATE_ARGS
#     "
# TO:
# cmd="torchrun $DISTRIBUTED_ARGS $SCALEHOME/pretrain_gpt.py \
#               $TRAINING_ARGS \
#               $MIXED_PRECISION_ARGS \
#               $DATA_ARGS \
#               $NETWORK_ARGS \
#               $INITIALIZATION_ARGS \
#               $REGULARIZATION_ARGS \
#               $LEARNING_RATE_ARGS \
#               $LOG_ARGS
#     "
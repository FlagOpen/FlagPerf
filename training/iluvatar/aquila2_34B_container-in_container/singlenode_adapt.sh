VOCAB_FILE=$SCALEHOME/examples/aquila/tokenizer/vocab.json
MERGE_FILE=$SCALEHOME/examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$SCALEHOME/examples/aquila/tokenizer/special_tokens.txt

TB_PATH=./aquila34B_perfermance
mkdir -p $TB_PATH

DISTRIBUTED_ARGS="
    --nproc_per_node 16 \
    --nnodes $NNODES \
    --node_rank $NODERANK \
    --master_addr $MASTERADDR \
    --master_port $MASTERPORT
"

TRAINING_ARGS="
    --mlp-g1-tp-overlap \
    --mlp-g2-tp-overlap \
    --attn-g1-tp-overlap \
    --attn-g2-tp-overlap \
    --mlp-g1-overlap-size 4 \
    --mlp-g2-overlap-size 4 \
    --attn-g1-overlap-size 4 \
    --attn-g2-overlap-size 4 \
    --mlp-g1-save-total-input-for-backward \
    --attn-g1-save-total-input-for-backward \
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --eval-interval 2000 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --use-flash-attn \
    --num-layers-per-virtual-pipeline-stage 5 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-method-per-stage 11 0 1 1 \
    --recompute-num-layers-per-stage 11 1 1 1 \
    --sequence-parallel
"

DATA_ARGS="
    --data-path $DATADIR/$DATASET \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --no-global-file-system
"

NETWORK_ARGS="
    --num-layers 60 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --hidden-dim-multiplier 1.3 \
    --swiglu \
    --multiple-of 4096\
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --layernorm-epsilon 1e-5 \
    --layernorm-init-weight 0.3 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --apply-layernorm-rms \
    --make-vocab-size-divisible-by 64 \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 42 
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
# for iluvatar performance testing, we need to change corresponding singlenode_run.sh :
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
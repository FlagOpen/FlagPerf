VOCAB_FILE=$SCALEHOME/examples/aquila/tokenizer/vocab.json
MERGE_FILE=$SCALEHOME/examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=$SCALEHOME/examples/aquila/tokenizer/special_tokens.txt

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
    --sequence-parallel \
    --use-flash-attn \
    --no-global-file-system \
    --recompute-granularity full \
    --recompute-method uniform 
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1.0 \
    --loss-scale-window 1024 \
    --attention-softmax-in-fp32\
    --embedding-weights-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

NETWORK_ARGS="
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --hidden-dim-multiplier 1.3 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --layernorm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-in-fp32 \
    --no-position-embedding \
    --swiglu \
    --multiple-of 4096 \
    --apply-layernorm-rms \
    --make-vocab-size-divisible-by 64 \
    --untie-embeddings-and-output-weights \
    --standalone-embedding-stage \
    --num-layers-of-first-stage 1 
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 42 
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
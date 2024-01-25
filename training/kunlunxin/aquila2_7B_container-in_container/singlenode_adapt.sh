
TRAINING_ARGS="
    --train-iters $TRAININGSAMPLES/$GBS \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --no-gradient-accumulation-fusion \
    --disable-bias-linear \
    --sequence-parallel \
    --distributed-backend nccl \
    --use-flash-attn
"

MIXED_PRECISION_ARGS="
    --embedding-weights-in-fp32 \
    --rotary-position-embeddings-in-fp32 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

LEARNING_RATE_ARGS="
    --lr 0.0003 \
    --min-lr 1.0e-5 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --lr-warmup-fraction .01 \
"

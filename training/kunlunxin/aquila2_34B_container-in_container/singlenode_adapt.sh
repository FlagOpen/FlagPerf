TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --log-interval 1 \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --sequence-parallel \
    --no-gradient-accumulation-fusion \
    --distributed-backend nccl
"

MIXED_PRECISION_ARGS="
    --embedding-weights-in-fp32 \
    --rotary-position-embeddings-in-fp32 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

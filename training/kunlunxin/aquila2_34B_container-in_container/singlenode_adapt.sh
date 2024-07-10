
TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
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
    --fp16 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1.0 \
"

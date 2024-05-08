TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --bf16 \
    --fp8-format hybrid \
    --fp8-amax-compute-algo max \
    --fp8-amax-history-len 1024
"

echo "[Prompt] kunlunxin adaption start"

TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --fp16 \
    --initial-loss-scale 16384
"

VENDOR_ARGS=" \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer \
    --use-mcore-models \
    --use-flash-attn \
    --disable-bias-linear \
    --use-cpu-initialization --hidden-dropout 0 --attention-dropout 0 \
    --no-async-tensor-model-parallel-allreduce --no-gradient-accumulation-fusion
"


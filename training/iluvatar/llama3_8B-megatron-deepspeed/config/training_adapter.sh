export CUDA_DEVICE_MAX_CONNECTIONS=1
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1

TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --bf16 \
    --transformer-impl local \
    --eval-iters 0 \
    --disable-bias-linear \
    --eval-interval 100 \
    --no-fp8-wgrad \
    --custom-partition 4 4 4 4 4 4 5 3 \
    --recompute-granularity full \
    --recompute-method block \
    --custom-recompute-layers-per-stage 3 2 2 1 0 0 0 0 \
    --no-load-optim \
    --no-load-rng \
    --initial-loss-scale 4096 \
    --min-loss-scale 1.0 \
    --no-query-key-layer-scaling \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --untie-embeddings-and-output-weights \
    --rotary-position-embeddings-theta 500000 \
    --make-vocab-size-divisible-by 16032 \
    --seed 1234 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --lr-warmup-iters 0 \
    --save-interval 10000
"
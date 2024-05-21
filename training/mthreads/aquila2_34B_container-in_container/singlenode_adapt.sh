# CHECKPOINT_PATH=/home/dist/zhiyuan-test/checkpoint

TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --no-gradient-accumulation-fusion \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --distributed-backend mccl \
    --use-flash-attn \
    --sequence-parallel \
    --device-type mthreads
 "

# --recompute-granularity full \
# --recompute-method block \
# --recompute-num-layers 11
MIXED_PRECISION_ARGS="
    --fp16 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1.0 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --rotary-position-embeddings-in-fp32 \
    --log-interval 1
"

LEARNING_RATE_ARGS="
    --lr 1.5e-6 \
    --min-lr 1.5e-7 \
    --lr-decay-style cosine \
    --lr-warmup-samples 40960
"
#for 48 gpus
#--lr-warmup-samples 7680 

# for 256 gpus
# --lr-warmup-samples 40960
# --lr-warmup-samples = 10(wamrup_step) * 4096(gbs)

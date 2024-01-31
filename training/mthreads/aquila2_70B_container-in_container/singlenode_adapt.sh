CHECKPOINT_PATH=$SCALEHOME/aquila/checkpoints/$TP-$PP-$MBS-$GBS-$TRAININGSAMPLES
mkdir -p $CHECKPOINT_PATH

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
    --device-type mthreads \
    --recompute-num-layers 5
"
# --recompute-granularity full \
# --recompute-method block \
# --recompute-num-layers 0

MIXED_PRECISION_ARGS="
    --fp16 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1.0 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --rotary-position-embeddings-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
"

LEARNING_RATE_ARGS="
    --lr 1.1e-6 \
    --min-lr 1.1e-7 \
    --lr-decay-style cosine \
    --lr-warmup-samples 4853760 \
    --log-interval 1 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --save-interval 10
"
# for 1024 gpus
# --recompute-num-layers 6 
# --save $CHECKPOINT_PATH \
# --load $CHECKPOINT_PATH \
# --save-interval 10

# for 256 gpus
# --recompute-num-layers 4

# for 128 gpus
# --recompute-num-layers 3

# --lr-warmup-samples 7680 #for 128 gpus warmup step 5
# --lr-warmup-samples 15360 #for 256 gpus warmup step 5
# --lr-warmup-samples 4853760 #for 1024 gpus warmup step 395

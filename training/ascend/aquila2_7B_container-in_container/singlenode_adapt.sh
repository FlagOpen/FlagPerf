TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --no-shared-fs \
    --no-gradient-accumulation-fusion \
    --use-flash-attn \
    --npu-fa-pre-tokens 2048 \
    --npu-fa-next-tokens 0 \
    --npu-fa-shape-order SBH \
    --use-npu-swiglu \
    --make-vocab-size-divisible-by 8 \
    --device-type ascend \
    --log-interval 1 
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --data-path $DATADIR/$DATASET \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008 \
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --data-impl mmap \
    --split 1 \
    --distributed-timeout-minutes 120
"

LEARNING_RATE_ARGS="
    --lr 2.0e-5 \
    --min-lr 2.0e-6 \
    --lr-decay-style cosine \
    --lr-warmup-samples 7200
"
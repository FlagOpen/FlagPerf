TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --tensor-model-parallel-size 8 \
    --sequence-parallel \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --no-shared-fs \
    --no-gradient-accumulation-fusion \
    --use-flash-attn \
    --pre-tokens 2048 \
    --next-tokens 0 \
    --shape-order SBH \
    --use-npu-mc2 \
    --use-npu-swiglu
"

MIXED_PRECISION_ARGS="
    --bf16 \
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
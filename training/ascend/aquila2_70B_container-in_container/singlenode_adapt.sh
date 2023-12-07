TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --sequence-parallel \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --no-shared-fs \
    --use-npu-swiglu \
    --use-flash-attn \
    --pre-tockens 65536 \
    --next-tockens 0 \
    --shape-order SBH
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --embedding-weights-in-fp32 \
    --rotary-position-embeddings-in-fp32 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --data-path $DATASETDIR \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008 \
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --data-impl mmap \
    --split 1 \
    --distributed-timeout-minutes 120
"
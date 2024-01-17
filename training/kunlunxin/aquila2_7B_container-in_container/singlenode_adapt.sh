LOAD_CHECKPOINT_PATH=/XMLIR/checkpoints/load_aiplat
echo "LOAD_CHECKPOINT_PATH", $LOAD_CHECKPOINT_PATH

    #--train-samples 48828 \
    #--tensorboard-dir ./aquila7b_tensorboard \
    #--load $LOAD_CHECKPOINT_PATH
TRAINING_ARGS="
    --train-samples $TRAININGSAMPLES \
    --log-interval 1 \
    --tensorboard-log-interval 1 \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --no-gradient-accumulation-fusion \
    --disable-bias-linear \
    --sequence-parallel \
    --distributed-backend nccl
 "

    #--train-iters 5 \
    #--use-distributed-optimizer \
MIXED_PRECISION_ARGS="
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

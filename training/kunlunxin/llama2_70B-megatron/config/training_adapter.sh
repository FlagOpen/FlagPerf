export PYTHONPATH=$PYTHONPATH:/home/FlagScale

MIXED_PRECISION_ARGS=""

CODE_PATH="/home/FlagScale/pretrain_llama.py"

TRAINING_ARGS="
    --train-samples $TRAIN_SAMPLES \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --micro-batch-size $M_BATCHSIZE \
    --global-batch-size $G_BATCHSIZE \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --recompute-granularity 'full' \
    --recompute-num-layers 1 \
    --recompute-method 'uniform' \
    --no-async-tensor-model-parallel-allreduce \
    --distribute-saved-activations
"
NETWORK_ARGS="
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --ffn-hidden-size 28672 \
    --seq-length $SEQLENGTH \
    --max-position-embeddings $SEQLENGTH \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 4096 \
    --untie-embeddings-and-output-weights
"


export BKCL_CCIX_BUFFER_GM=1
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1

export BKCL_SOCKET_IFNAME=ibs11
export BKCL_USE_RDMA=0

export BKCL_RDMA_FORCE_TREE=1
export BKCL_ENABLE_XDR=0
export BKCL_RING_BUFFER_SIZE=1024000
export BKCL_RDMA_NICS=ibs11
export BKCL_FORCE_ALLREDUCE_IN_MULTINODE=1
worker_num=0

ulimit -c 0
export XMLIR_F_XPU_ENABLED_BOOL=true
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=1800
export BKCL_FORCE_SYNC=1


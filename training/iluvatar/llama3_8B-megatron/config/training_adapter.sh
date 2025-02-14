echo "[Prompt] iluvatar adaption is not NULL, for other Vendors"
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=4
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
VENDOR_ARGS=" \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --no-create-attention-mask-in-dataloader \
    --use-legacy-models \
    --num-layers-per-stage 1 7 2 9 1 7 \
    --disable-bias-linear \
"

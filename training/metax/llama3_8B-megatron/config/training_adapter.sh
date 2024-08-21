# for 1*8 3150
VENDOR_ARGS=" \
    --transformer-impl local  \
    --use-distributed-optimizer \
    --use-mcore-models \
    --use-flash-attn \
    --pipline-num-layers-list 7 9 9 7
"
# for 4*8
# VENDOR_ARGS=" \
#     --transformer-impl local  \
#     --use-distributed-optimizer \
#     --use-mcore-models \
#     --use-flash-attn \
#     --attention-dropout 0.0 \
#     --hidden-dropout 0.0 \
#     --recompute-granularity full \
#     --recompute-method block \
#     --recompute-num-layers 1 \
#     --recompute-num-layers-list 2 0
# "
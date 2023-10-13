#!/bin/bash
my_dir=$1
pip install jsonlines
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=enp
export NCCL_IB_DISABLE=1
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_0,mlx5_3
export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4
#export CUDA_VISIBLE_DEVICE=1,2
WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --master_port 6000"

TASK="AQUILA"

VALID_DATA="$my_dir/data/lambada_test_bak.jsonl"
VOCAB_FILE="$my_dir/tokenizer/vocab.json"
MERGE_FILE="$my_dir/tokenizer/merges.txt"
CHECKPOINT="$my_dir/ckpts/"
SPECIAL_TOKENS_FILE="$my_dir/tokenizer/special_tokens.txt"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task $TASK \
               --eval-metric loss \
               --valid-data $VALID_DATA \
    	       --tokenizer-type AquilaTokenizer \
               --vocab-file $VOCAB_FILE \
               --merge-file $MERGE_FILE \
	       --special-tokens-file $SPECIAL_TOKENS_FILE  \
               --load $CHECKPOINT \
	       --tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --num-layers 60 \
               --hidden-size 6144 \
               --num-attention-heads 48 \
               --group-query-attention \
               --num-query-groups 8 \
               --hidden-dim-multiplier 1.3 \
               --seq-length 4096 \
               --max-position-embeddings 4096 \
               --layernorm-epsilon 1e-5 \
               --layernorm-init-weight 0.3 \
               --use-rotary-position-embeddings \
               --no-position-embedding \
               --swiglu \
               --multiple-of 4096 \
               --apply-layernorm-rms \
               --untie-embeddings-and-output-weights \
               --disable-bias-linear \
               --log-interval 1 \
               --bf16 \
               --make-vocab-size-divisible-by 64 \
               --micro-batch-size 1 \
               --no-load-optim \
               --no-load-rng

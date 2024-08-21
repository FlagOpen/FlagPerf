export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM-FlagScale
CODE_PATH="/workspace/Megatron-LM-FlagScale/pretrain_llama.py"
RECOMPUTE_ARGS="
    --pipline-num-layers-list 9 9 10 10 10 11 11 10
"
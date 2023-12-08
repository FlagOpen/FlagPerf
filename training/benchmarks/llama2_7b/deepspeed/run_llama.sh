set -x
export XDNN_FC_GEMM_DTYPE="float32"

deepspeed --num_gpus=8 run_pretraining.py --deepspeed --deepspeed_config ds_config.json --data_dir ./data  --flagperf_config  ./config/config_A100x1x8.py --nproc 8 --nnodes 1

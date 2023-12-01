set -x
#export XACC=1
#export  IM_XACC_DEVELOPER=1
#export XLOG_LEVEL='info,capture=info'
#export XPUAPI_DEBUG=0x1
#export XPURT_DISPATCH_MODE=PROFILING
#export XACC_ARGS="-L ms_deepspeed"

deepspeed --num_gpus=8 run_pretraining.py --deepspeed --deepspeed_config ds_config.json --data_dir ./data  --flagperf_config  ./config/config_A100x1x8.py --nproc 8 --nnodes 1


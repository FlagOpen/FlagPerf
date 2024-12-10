export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL128
export NCCL_ALGO=Ring

mpirun --allow-run-as-root -np 8 all_reduce_perf -t 1 -g 1 -b 2M -e 32M -d float -o sum

export MCCL_PROTOS=2
export MCCL_TOPO_ENHANCE_PLUGIN=None
export MCCL_ALGOS=1
export MUSA_BLOCK_DISTRIBUTION_GRANULARITY=1
export MUSA_EXECUTE_COUNT=1
export MCCL_BUFFSIZE=41943040
export MCCL_IB_GID_INDEX=3
mcc bandwidth.mu -o bdtest -lmusart -lmccl --offload-arch=mp_31
./bdtest

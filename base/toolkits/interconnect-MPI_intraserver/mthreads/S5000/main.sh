export MCCL_PROTOS=2
export MCCL_TOPO_ENHANCE_PLUGIN=None
mcc bandwidth.mu -o bdtest -lmusart -lmccl --offload-arch=mp_31
./bdtest

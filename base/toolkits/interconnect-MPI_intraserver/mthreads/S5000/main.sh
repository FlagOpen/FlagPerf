export MCCL_PROTOS=2
mcc bandwidth.mu -o bdtest -lmusart -lmccl --offload-arch=mp_31
./bdtest

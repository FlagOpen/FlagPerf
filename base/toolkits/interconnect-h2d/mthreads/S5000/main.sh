export MUSA_MEMCPY_PATH=3 
mcc bandwidth.mu -o bdtest -lmusart --offload-arch=mp_31
./bdtest

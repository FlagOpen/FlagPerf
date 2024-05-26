#!/bin/bash
service ssh start
export NCCL_DEBUG=WARN
nvcc -c -o bandwidth.o bandwidth.cu -I/usr/local/cuda/include -I/usr/local/nccl/include -I/usr/local/mpi/include
mpic++ -o bdtest bandwidth.o -L/usr/local/cuda/lib64 -L/usr/local/nccl/lib -L/usr/local/mpi/lib -lcudart -lnccl -lcuda -lmpi
echo "NODERANK: $NODERANK"
if [ "$NODERANK" -eq 0 ]; then
    echo "NODERANK is 0, executing the final command..."
    sleep 10
    mpirun --allow-run-as-root --host 10.1.2.155,10.1.2.158 -np 2 -x NCCL_DEBUG=WARN -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_2,mlx5_5 -x CUDA_DEVICE_MAX_CONNECTIONS=1 ./bdtest
fi

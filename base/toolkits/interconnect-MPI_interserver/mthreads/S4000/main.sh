#!/bin/bash
service ssh restart
export MCCL_DEBUG=WARN
export MCCL_PROTOS=2
export OPAL_PREFIX=/opt/hpcx/ompi
export PATH=/opt/hpcx/ompi/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib/:/usr/local/musa/lib/:$LD_LIBRARY_PATH
export MUSA_KERNEL_TIMEOUT=3600000
HOSTS=$(yq '.HOSTS | map(. + ":8") | join(",")' ../../../../configs/host.yaml)
mcc -c -o bandwidth.o bandwidth.mu -I/usr/local/musa/include -I/opt/hpcx/ompi/include -fPIC
mpic++ -o bdtest bandwidth.o -L/usr/local/musa/lib -lmusart -lmccl -lmusa -lmpi
echo "NODERANK: $NODERANK"
if [ "$NODERANK" -eq 0 ]; then
    echo "NODERANK is 0, executing the final command..."
    sleep 10
    mpirun --allow-run-as-root --host $HOSTS -x MCCL_PROTOS=2 -x MCCL_DEBUG=WARN -x MCCL_IB_DISABLE=0 -x MCCL_IB_HCA=mlx5_0,mlx5_1 -x MUSA_DEVICE_MAX_CONNECTIONS=1 ./bdtest
fi
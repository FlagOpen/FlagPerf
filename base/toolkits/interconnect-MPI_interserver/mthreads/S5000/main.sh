#!/bin/bash
service ssh restart
export MCCL_PROTOS=2
export MUSA_EXECUTION_TIMEOUT=1000000
export MCCL_GRAPH_FILE=/usr/local/musa/topo
export OMP_PATH=/usr/local/openmpi/
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$OMP_PATH/local/musa/lib/:$LD_LIBRARY_PATH
HOSTS=$(yq '.HOSTS | map(. + ":8") | join(",")' ../../../../configs/host.yaml)
mcc -c -o bandwidth.o bandwidth.mu -I/usr/local/musa/include -I/usr/local/openmpi/include -fPIC --offload-arch=mp_31
mpic++ -o bdtest bandwidth.o -L/usr/local/musa/lib -lmusart -lmccl -lmusa -lmpi
echo "NODERANK: $NODERANK"
if [ "$NODERANK" -eq 0 ]; then
    echo "NODERANK is 0, executing the final command..."
    sleep 10
    mpirun  --host $HOSTS \
            --mca btl_tcp_if_include ens19f0np0 \
            --allow-run-as-root \
            --map-by ppr:8:node \
	        -oversubscribe \
            -x MCCL_ALGOS=1 \
            -x MUSA_BLOCK_DISTRIBUTION_GRANULARITY=1 \
            -x MUSA_EXECUTE_COUNT=1 \
            -x MCCL_BUFFSIZE=41943040 \
            -x MUSA_EXECUTION_TIMEOUT=1000000 \
            -x MCCL_IB_TIMEOUT=22 \
            -x MCCL_IB_GID_INDEX=3 \
            -x MCCL_TOPO_ENHANCE_PLUGIN=None \
            -x MCCL_PROTOS=2 \
            -x MUSA_DEVICE_MAX_CONNECTIONS=1 \
            -x MCCL_GRAPH_FILE=/usr/local/musa/topo \
            -x MCCL_IB_HCA='=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1'
            ./bdtest
fi
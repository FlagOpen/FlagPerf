MASTERADDR=$1
MASTERPORT=$2
NPROC=$3
NODE=$4
NODERANK=$5
torchrun --nproc_per_node $NPROC --nnodes $NODE --node_rank $NODERANK --master_addr $MASTERADDR --master_port $MASTERPORT global.py
torchrun --nproc_per_node $NPROC --nnodes $NODE --node_rank $NODERANK --master_addr $MASTERADDR --master_port $MASTERPORT server.py
torchrun --nproc_per_node $NPROC --nnodes $NODE --node_rank $NODERANK --master_addr $MASTERADDR --master_port $MASTERPORT prime.py

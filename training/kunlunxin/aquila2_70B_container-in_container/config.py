SSH_PORT = "22"

flops_16bit = "128000000000000.0"

#multi_node communication
net_cmd ="export BKCL_CCIX_RING=1;export BKCL_CCIX_BUFFER_GM=1;export BKCL_TREE_THRESHOLD=1;export BKCL_TIMEOUT=1800;export \
BKCL_SOCKET_IFNAME=ibs11;export CUDA_DEVICE_MAX_CONNECTIONS=1"

env_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

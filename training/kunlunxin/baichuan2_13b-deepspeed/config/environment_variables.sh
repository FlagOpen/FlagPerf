export LD_LIBRARY_PATH=/workspace/bkcl_so/so:/workspace/xre-Linux-x86_64-0.0.0.1-2024-03-28-23-30-24-daily/so:$LD_LIBRARY_PATH

export CUDART_DUMMY_REGISTER=1
export XPU_DUMMY_EVENT=1

# ulimit -c 0
export NCCL_SOCKET_IFNAME=xgbe0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

#################################
# driver方式部分
#################################
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export XPU_DUMMY_EVENT=1

#################################
# 算子部分
#################################
export XPUAPI_DEFAULT_SIZE=4000000000
export XBLAS_FC_HBM_VERSION=40

#################################
# 算子检查部分
#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

#################################
# hbm部分
#################################
export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1

#################################
# BKCL C2C部分
#################################
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1

# ccix_inner_8chips
cat > ccix_inter.txt <<EOF
[chip 0, port 0] <===> [chip 6, port 3]
[chip 4, port 1] <===> [chip 5, port 2]
[chip 2, port 3] <===> [chip 4, port 0]
[chip 2, port 1] <===> [chip 3, port 2]
[chip 1, port 1] <===> [chip 3, port 1]
[chip 0, port 1] <===> [chip 1, port 2]
[chip 0, port 2] <===> [chip 2, port 2]
[chip 0, port 3] <===> [chip 3, port 3]
[chip 1, port 0] <===> [chip 2, port 0]
[chip 1, port 3] <===> [chip 7, port 0]
[chip 5, port 1] <===> [chip 7, port 1]
[chip 3, port 0] <===> [chip 5, port 3]
[chip 4, port 3] <===> [chip 7, port 3]
[chip 4, port 2] <===> [chip 6, port 2]
[chip 6, port 1] <===> [chip 7, port 2]
[chip 5, port 0] <===> [chip 6, port 0]
EOF

export XPU_ZEBU_MODE=1
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_RING_BUFFER_GM=1
export BKCL_FORCE_SYNC=1
export BKCL_KL3_TURBO_MODE=1 
export BKCL_RING_BUFFER_SIZE=2097152
export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=3600
unset BKCL_KL3_SYSCON_FLAG

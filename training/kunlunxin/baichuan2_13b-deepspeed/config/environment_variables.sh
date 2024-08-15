#################################
# driver方式部分
#################################
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export XPU_DUMMY_EVENT=1

#################################
# BKCL C2C部分
#################################
export BKCL_CCIX_BUFFER_GM=1
export BKCL_CCIX_RING=1
export BKCL_ENABLE_XDR=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_FORCE_SYNC=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_RDMA_NICS=ens11np0,ens11np0,ens13np0,ens13np0,ens15np0,ens15np0,ens17np0,ens17np0
export BKCL_RING_BUFFER_GM=1
export BKCL_RING_BUFFER_SIZE=2097152
export BKCL_SOCKET_IFNAME=ens21f0np0
export BKCL_TIMEOUT=360000
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_TREE_THRESHOLD=1
export BKCL_XLINK_C2C=1
export BKCL_XLINK_D2D=0
export BKCL_XLINK_ETH=0
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0

# 性能
export XDNN_USE_FAST_SWISH=true
export XDNN_FAST_DIV_SCALAR=true
export XMLIR_BMM_DISPATCH_VALUE=2
export XBLAS_FC_HBM_VERSION=40
export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1
export XPUAPI_DEFAULT_SIZE=4000000000

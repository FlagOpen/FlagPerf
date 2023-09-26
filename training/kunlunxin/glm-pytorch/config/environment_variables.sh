# =================================================
# Export variables
# =================================================

export BKCL_PCIE_RING=1
export BKCL_TIMEOUT=1800
# when using tree allreduce, the number of nodes must be a multiple of 2
export BKCL_SOCKET_FORCE_TREE=1

export XMLIR_D_XPU_L3_SIZE=32505856

export BKCL_CCIX_RING=1
export BKCL_FORCE_SYNC=1

export ALLREDUCE_ASYNC=false
export ALLREDUCE_FUSION=0

export XMLIR_F_XPU_FC_GEMM_MODE=float
export XMLIR_F_FAST_INDEX_PUT=true

export XACC_ENABLE=1

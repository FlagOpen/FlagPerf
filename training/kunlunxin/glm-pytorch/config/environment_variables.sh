# =================================================
# Export variables
# =================================================

export BKCL_PCIE_RING=1
export BKCL_TIMEOUT=1800
# when using tree allreduce, the number of nodes must be a multiple of 2
export BKCL_SOCKET_FORCE_TREE=1

export XMLIR_D_XPU_L3_SIZE=66060288

export ALLREDUCE_FUSION=0

export XACC=1
export XACC_ARGS="-L O0"

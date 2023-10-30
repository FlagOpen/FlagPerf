export XACC_ENABLE=1
export BKCL_PCIE_RING=1
export XMLIR_D_FORCE_FALLBACK_STR="aten::index_add_"
export XMLIR_FALLBACK_OP_LIST_FILE_PATH="log"
export XMLIR_XPU_EAGER_LAUNCH_SYNC_MODE=true

python3 -m xacc.install

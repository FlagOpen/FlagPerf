export XACC_ENABLE=1
export BKCL_PCIE_RING=1
export XMLIR_D_XPU_L3_SIZE=66060288
export XMLIR_D_FORCE_FALLBACK_STR="aten::_index_put_impl_,aten::index.Tensor"

export XACC_ARGS="-L auto_tune"

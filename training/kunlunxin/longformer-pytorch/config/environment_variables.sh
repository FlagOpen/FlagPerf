export XACC_ENABLE=1
export BKCL_PCIE_RING=1
export XMLIR_D_FORCE_FALLBACK_STR="aten::index_add_"
export XMLIR_FALLBACK_OP_LIST_FILE_PATH="log"
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true
export XMLIR_XPU_EAGER_LAUNCH_SYNC_MODE=true
#export XLOG_LEVEL="capture=info"
#export XMLIR_D_FILE_LOG_BOOL=1

pip install /data/GGuan/FlagPerf/training/dist/xacc-0.1.0-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install /data/GGuan/FlagPerf/training/dist/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m xacc.install

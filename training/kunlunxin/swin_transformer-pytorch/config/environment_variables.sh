# =================================================
# Export variables
# =================================================

export XMLIR_F_XPU_ENABLED_BOOL=true
export XMLIR_TORCH_XCCL_ENABLED=true


# =================================================
# R480 config
# =================================================

export OMP_NUM_THREADS=1
export XACC_ARGS="-L amp"
export XACC=1
export BKCL_PCIE_RING=1

KLX_WEB_SERVER_URL=http://127.0.0.1:8000

pip uninstall -y xacc || true
pip install ${KLX_WEB_SERVER_URL}/flagperf/archives/9bb59e9e/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
pip uninstall -y xmlir || true
pip install ${KLX_WEB_SERVER_URL}/flagperf/archives/9bb59e9e/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl

python -m xacc.install
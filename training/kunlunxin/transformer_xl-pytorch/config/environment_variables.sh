export XACC_ENABLE=1
export BKCL_PCIE_RING=1

pip install /data/GGuan/FlagPerf/training/dist/xacc-0.1.0-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install /data/GGuan/FlagPerf/training/dist/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m xacc.install

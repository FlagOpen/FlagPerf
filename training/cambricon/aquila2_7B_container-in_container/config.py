# =========================================================
# network
# =========================================================
SSH_PORT = "22"

net_cmd = "export MLU_DEVICE_MAX_CONNECTIONS=1"

# =========================================================
# chip attribute
# =========================================================
flops_16bit = "294900000000000"

# =========================================================
# env attribute
# =========================================================
env_cmd = "export LD_LIBRARY_PATH=/torch/neuware_home/lib64;export NEUWARE_HOME=/torch/neuware_home"

'''Cluster configs'''

# Hosts to run the benchmark. Each item is an IP address or a hostname.
HOSTS = ["10.1.2.2", "10.1.2.3", "10.1.2.4"]

# Hosts port to run the tensorflow distribution_strategy = 'multi_worker_mirrored'
HOSTS_PORTS = ["2222"]

# Master port to connect
MASTER_PORT = "29501"

# ssh connection port
SSH_PORT = "22"
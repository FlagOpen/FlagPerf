import os
import subprocess

if __name__ == "__main__":

    iplist = []
    for line in open("hostfile").readlines():
        iplist.append(line.replace("\n", ""))

    master_addr = iplist[0]
    master_port = "29501"
    nproc_per_node = "8"
    nnodes = len(iplist)

    noderank = 0
    procs = []
    for ip in iplist:

        f = open(ip + ": node" + str(noderank) + ".log.txt", "w")
        path = os.path.dirname(os.path.abspath(__file__))

        exec_cmd = "cd " + path + ";source env.sh;bash server.sh " + master_addr + " " + master_port + " " + nproc_per_node
        exec_cmd = exec_cmd + " " + str(nnodes) + " " + str(noderank)

        exec_cmd = "\"" + exec_cmd + "\""

        ssh_exec_cmd = ["ssh", ip, exec_cmd]
        exec_cmd = ' '.join(ssh_exec_cmd)

        print(exec_cmd)

        p = subprocess.Popen(exec_cmd,
                             shell=True,
                             stdout=f,
                             stderr=subprocess.STDOUT)
        procs.append((p, f))

        noderank += 1

    for p, f in procs:
        p.wait()
        f.close()
